#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// ROS
#include <image_geometry/pinhole_camera_model.h>

// kitti
#include <kitti/common/serialization/dataset.h>

// image_cloud hpp
#include <image_cloud/common/filter/cv/inverse_distance_transform.hpp>
#include <image_cloud/common/filter/cv/edge.hpp>
#include <image_cloud/common/transform.hpp>
#include <image_cloud/common/calibration/score.hpp>
#include <image_cloud/common/calibration/structs.hpp>
#include <image_cloud/common/calibration/grid_search.hpp>
#include <image_cloud/common/calibration/pipeline/image.hpp>
#include <image_cloud/common/calibration/pipeline/pointcloud.hpp>
#include <image_cloud/common/calibration/pipeline/enums.h>

#include <kitti_calibration/common/messure_time.h>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map opts;

void run_search(kitti::Dataset data, int camera, int sequence, std::vector<search::Search_value> &results) {

	image_geometry::PinholeCameraModel camera_model;
	data.camera_list.at(camera).get_camera_model(camera_model);

	std::deque<cv::Mat> list_images;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points;
	bool pre_filtred = true;

	for(int i = 0; i < opts["window"].as<int>(); ++i){
		// Load images
		cv::Mat image_load, image_inverse;
		data.camera_file_list.at(camera).load_image(image_load, sequence+i);

		// create inverse imageimage_pointcloud::
		image_cloud::create_inverse_transformed(image_load, image_inverse);
		list_images.push_back(image_inverse);
		// Image processing done

		// Load Pointcloud
		pcl::PointCloud < pcl::PointXYZI > transformed;
		data.pointcloud_file_list.load_pointcloud(transformed, sequence+i);
		//
		//	Transform PointCloud
		tf::Transform velo_to_cam0,cam0_to_cam;

		data.velodyne_to_cam0.get_transform(velo_to_cam0);
		data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);

		tf::Transform tf_result = (cam0_to_cam * velo_to_cam0);

		image_cloud::transform_pointcloud(transformed, tf_result);

		pcl_filter::Filter3d filter = (pcl_filter::Filter3d)opts["filter"].as<int>();

		// Filter pointclouds once for all TFs
		if(pre_filtred){
			pcl::PointCloud < pcl::PointXYZI > points_filtred;
			image_cloud::filter3d_switch <pcl::PointXYZI > (transformed, points_filtred, camera_model, camera, sequence, filter, image_load.rows, image_load.cols);
			list_points.push_back(points_filtred);
		}
		else{
			list_points.push_back(transformed);
		}
	}
	search::calculate<pcl::PointXYZI,uchar>(camera_model, list_points, list_images, results, pre_filtred );
}

int main(int argc, char* argv[]){
	std::stringstream available_pcl_filters;

	available_pcl_filters << "pcl filter: \n";
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		available_pcl_filters << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n ";
	}
	std::vector<float> range;
	std::vector<float> start;
	std::vector<int> steps;
	range.resize(6);
	start.resize(6);
	steps.resize(6);

	for (int i = 0; i < 6; ++i){
		steps[i] = 3;
		range[i] = 0.01;
		start[i] = 0;
	}
	//std::vector<float>() range {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
	po::options_description desc("Usage");
	desc.add_options()
	("help", "Print this help messages")
	("i", po::value<std::string>()->default_value("/media/Daten/kitti/kitti/2011_09_26_drive_0005_sync/"), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera nr")
	("start", po::value<int>()->default_value(0), "sequence number")
	("window", po::value<int>()->default_value(1), "sliding window size")
	("range", po::value< std::vector<float> >(&range)->multitoken(), "search range: x y z roll pitch yaw")
	("steps", po::value< std::vector<int> >(&steps)->multitoken(), "steps: x y z roll pitch yaw")
	("start_transform", po::value< std::vector <float > >(&start)->multitoken(), "start transforme: x y z roll pitch yaw")
	("save_images", po::value<bool>()->default_value(true), "write image files")
	("f", po::value<bool>()->default_value(false), "process full kitti dataset")
	("p", po::value<std::string>()->default_value(""), "output file prefix")
	("iterations", po::value<int>()->default_value(1), "optimation iterration (halves search range per iterration")
	("filter", po::value<int>()->default_value(pcl_filter::DEPTH_INTENSITY), available_pcl_filters.str().c_str())
	("out",	po::value<std::string>()->default_value(""),"calculation output file");

	//po::store(po::parse_command_line(argc, argv, desc), opts);
	po::store(parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), opts);


	try {
		po::notify(opts);

	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		std::cerr << desc << std::endl;
		return 1;
	}

    /** --help option
     */

	if ( opts.count("help")  )
	{
		std::cout 	<< "Kitti offline edge calibration" << std::endl
					<< desc << std::endl;
		return 0;
	}

	tf::Transform search_startpoint;
	tf::Quaternion q;
	search_startpoint.setIdentity();
	search_startpoint.setOrigin(tf::Vector3(start[0], start[1], start[2]));
	q.setRPY(start[3], start[4], start[5]);
	search_startpoint.setRotation(q);

	kitti::Dataset data(opts["i"].as<std::string>());

	std::vector<search::Search_setup> search_configs;
	std::vector<search::Multi_search_result> search_results;
	float factor = 2;
	//search_startpoint;

	std::ofstream myfile;
	std::string filename = opts["out"].as<std::string>();
	if(!filename.empty()){
		myfile.open(filename.c_str());
	}

	clock_t clock_start = clock();
	for(int i = 0; i < opts["iterations"].as<int>(); ++i){

		std::stringstream out;
		search::Multi_search_result multi_result;
		std::vector<search::Search_value> results;
		tf::Transform tmp;
		tf::Transform best_until_now = search_startpoint;

		factor = factor / 2;

		// resize ranges
		std::cout << "factor * oldrange = range \t steps \t start \n";
		for(int j = 0; j<6 ;++j){
			std::cout << factor << " * " << range[j] << " = ";
			range[j] = range[j]*factor;
			std::cout << range[j] << "\t"
					  << steps[j] << "\t"
					  << start[j] <<"\n";
		}



		// find best tf for next search
		unsigned int score = 0;
		for(int j = 0; j < search_results.size(); ++j)
		{
			if(score < search_results.at(j).best.score){
				score = search_results.at(j).best.score;

				search_results.at(j).best.get_transform(best_until_now);
			}
		}

		//std::cout << score << "\t\n";

		// setup grid search
		search::Search_setup search_config(best_until_now, range, steps);
		//std::cout << "search setup" << search_config.to_string() << std::endl;

		search::grid_setup(search_config, results);

		// store current config
		search_configs.push_back(search_config);

		// run grid search
		run_search(data, opts["camera"].as<int>(), opts["start"].as<int>(), results);

		// evaluate result
		search::evaluate_results(results, tmp, &multi_result);

		// store results
		search_results.push_back(multi_result);


		// Output
		if(i == 0){
			out << "nr\t" << search_config.to_description_string() << "\t" << multi_result.to_description_string() << std::endl;
			std::cout << "nr\t" << search_config.to_description_string() << "\t" << multi_result.to_description_string() << std::endl;
		}
		out << i << "\t" << search_config.to_simple_string() << "\t" << multi_result.to_simple_string() << std::endl;
		std::cout << i << "\t" <<search_config.to_string() << "\t" << multi_result.to_string() << std::endl;

		if(!filename.empty()){
			myfile << out.str();
		}
	}
	clock_t clock_end = clock();


//	for(int i = 0; i < search_results.size(); ++i){
//		std::cout << i << ": " << search_results.at(i).best.to_simple_string() << std::endl;
//		for(int j = 0; j < search_results.at(i).best_results.size(); ++j){
//			std::cout << j << ":\tscore:" << search_results.at(i).best_results.at(j).score << std::endl;
//		}
//	}

	std::stringstream out;
	out << "\n";
	out << time_string(clock_start, clock_end, "global search", opts["iterations"].as<int>(),search_results.at(0).best_results.size(), search_results.at(search_results.size()-1).best.score, true);
	out << "\n\n\n";
	out << "nr" << "\t" << search_results.at(0).best.to_description_string() << "\tfc" << std::endl;
	for(int i = 0; i < search_results.size(); ++i){
		out
				<< i << "\t"
				<< search_results.at(i).best.to_simple_string() << "\t"
				<< search_results.at(i).get_fc() << "\t" << std::endl;
	}

	if(!filename.empty()){
		myfile << out.str();
		myfile.close();
	}

	return 0;
}
