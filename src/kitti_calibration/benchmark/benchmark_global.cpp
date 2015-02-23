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

#include <kitti_calibration/common/common.hpp>
#include <kitti_calibration/common/messure_time.h>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map opts;



//void pre_filter_data(const	std::deque<cv::Mat> &in_list_images,
//		const	std::deque<pcl::PointCloud<pcl::PointXYZI> > &in_list_points,
//		const image_geometry::PinholeCameraModel &camera_model,
//		const pcl_filter::Filter3d filter,
//		std::deque<cv::Mat> &out_list_images,
//		std::deque<pcl::PointCloud<pcl::PointXYZI> > &out_list_points
//		)
//{
//	assert(in_list_images.size() == in_list_points.size());
//
//	out_list_images.resize(in_list_images.size());
//	out_list_points.resize(in_list_images.size());
//
//	for(int i = 0; i < in_list_images.size(); ++i)
//	{
//		// Filter images
//		image_cloud::create_inverse_transformed(in_list_images.at(i), out_list_images.at(i));
//
//		// Filter Pointclouds
//		image_cloud::filter3d_switch <pcl::PointXYZI > (in_list_points.at(i), out_list_points.at(i), camera_model, filter, in_list_images.at(i).rows, in_list_images.at(i).cols);
//	}
//}

int main(int argc, char* argv[]){
	std::cout << "\nstarting..." << std::endl;

	std::vector<clock_t> timing;
	timing.push_back(clock());
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

	int iterations = 1;

	char labels[6] = {'x','y','z','r','p','y'};

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
	("seq", po::value<int>()->default_value(0), "sequence number")
	("window", po::value<int>()->default_value(1), "sliding window size")
	("range", po::value< std::vector<float> >(&range)->multitoken(), "search range: x y z roll pitch yaw")
	("steps", po::value< std::vector<int> >(&steps)->multitoken(), "steps: x y z roll pitch yaw")
	("tf", po::value< std::vector <float > >(&start)->multitoken(), "start transforme: x y z roll pitch yaw")
	("save_images", po::value<bool>()->default_value(true), "write image files")
	("f", po::value<bool>()->default_value(false), "process full kitti dataset")
	("p", po::value<std::string>()->default_value(""), "output file prefix")
	("iterations", po::value<int>(&iterations), "optimation iterration (halves search range per iterration")
	("filter", po::value<int>()->default_value(pcl_filter::DEPTH_INTENSITY), available_pcl_filters.str().c_str())
	("out",	po::value<std::string>()->default_value(""),"calculation output file")
	("blur",	po::value<bool>()->default_value(true),"blur images")
	("factor",	po::value<float>()->default_value(0.5),"reduce range step size per iteration using this factor")
	("precision", po::value<float>(), "if set, calibration runs until precision is reached")
	("pre_filter", po::value<bool>()->default_value(true), "apply point filter once, before search transformations");


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

	if( opts.count("precision") )
	{
		if(!(opts["factor"].as<float>() < 1)){
			std::cout << "ERROR precision cannot be reached. Factor needs to be below of 1.0 but set to: " << opts["factor"].as<float>() << std::endl;
			return 1;
		}

		iterations = 1;
		float tmp = range[0];

		while(tmp > opts["precision"].as<float>()){
			tmp = tmp*opts["factor"].as<float>();
			++iterations;
		}
		std::cout 	<< "\nInfo: Precision was set to: " << opts["precision"].as<float>() << ", set iterations to: " << iterations << std::endl;
	}

	search::Search_value best_result;
	best_result.init(start[0], start[1], start[2], start[3], start[4], start[5], 0);

	std::cout << "Opening KITTI dataset..." << spacer;
	timing.push_back(clock());

	kitti::Dataset data(opts["i"].as<std::string>());
	timing.push_back(clock());

	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	// Create buffers for search and results
	std::vector<search::Search_setup> search_configs;
	std::vector<search::Multi_search_result> search_results;
	std::vector<search::Search_value> search_chosen_tfs;

	// load and prepare kitti data
	std::deque<cv::Mat> list_images_raw,list_images_filtred;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_raw, list_points_filtred;
	image_geometry::PinholeCameraModel camera_model;

	std::cout << "Buffering data from KITTI dataset..." << spacer;
	std::cout.flush();
	timing.push_back(clock());
	load_kitti_data(data,
			list_images_raw,
			list_points_raw,
			camera_model,
			opts["camera"].as<int>(),
			opts["seq"].as<int>(),
			opts["window"].as<int>()
			);

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	// filter images
	std::cout << "pre filtering images" << spacer;
	pre_filter_images(list_images_raw, list_images_filtred, opts["blur"].as<bool>());

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	// Pre filter points
	std::cout << "pre filtering points:" << spacer;
	if(opts["pre_filter"].as<bool>()){
		pre_filter_points(list_points_raw, camera_model,
				(pcl_filter::Filter3d)opts["filter"].as<int>(),
				list_points_filtred,
				list_images_raw.at(0).rows,
				list_images_raw.at(0).cols
				);
	}

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	std::ofstream myfile;
	std::string filename = opts["p"].as<std::string>()+opts["out"].as<std::string>();
	if(!filename.empty()){
		myfile.open(filename.c_str());
	}

	std::cout << "started calibration...\n\n";
	clock_t clock_start = clock();

	// Calibration starts here
	for(int i = 0; i < iterations; ++i){
		timing.push_back(clock());
		std::stringstream out;

		search::Multi_search_result multi_result;
		std::vector<search::Search_value> results;


		// Create table for console output
		if(i==0){
			// print description
			search::Search_value search_desctiption;
			std::cout << "N" << spacer << search_desctiption.to_description_string() << spacer << "fc" << spacer << "total";
			for(int j = 0; j<6; ++j){
				std::cout << spacer << labels[j];
			}
			std::cout << spacer << "time";
			std::cout << std::endl;
		}
		else{
			// Print previous search result
			std::cout <<  i-1 << spacer
								<< best_result.to_simple_string() << spacer
								<< search_results.at(i-1).get_fc() << spacer
								<< search_results.at(i-1).best_results.size();

			// print old and calulate new range
			for(int j = 0; j<6 ;++j){
				std::cout << spacer << range[j] ;
				range[j] = range[j]*opts["factor"].as<float>();
			}

			// Print previous loop time
			std::cout << spacer << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1));
			std::cout << std::endl;
		}

		get_best_search_value(search_results, best_result);
		// setup grid search
		search::Search_setup search_config(best_result.get_transform(), range, steps);

		search::grid_setup(search_config, results);

		// store current config
		search_configs.push_back(search_config);

		// run grid search
		if(opts["pre_filter"].as<bool>()){
			search::calculate<pcl::PointXYZI,uchar>(camera_model, list_points_filtred, list_images_filtred, results, true);
		}
		else{
			search::calculate<pcl::PointXYZI,uchar>(camera_model, list_points_raw, list_images_filtred, results, false);
		}
		//run_search(data, opts["camera"].as<int>(), opts["start"].as<int>(), results);

		// evaluate result
		multi_result.evaluate(results);

		// store results
		search_results.push_back(multi_result);


		// Output
		if(i == 0){
			out << "nr\t" << multi_result.to_description_string() << "\t" << search_config.to_description_string() << std::endl;
			//std::cout << "nr\t" <<  multi_result.to_description_string() << "\t" << search_config.to_description_string() << std::endl;
		}
		out << i << "\t" << multi_result.to_simple_string() << "\t" << search_config.to_simple_string() << std::endl;
		//std::cout << i << "\t" << multi_result.to_string() << "\t" << search_config.to_string() << std::endl;

		if(!filename.empty()){
			myfile << out.str();
			myfile.flush();
		}

		// Create images of all best matching
		if(opts["save_images"].as<bool>()){
			tf::Transform tf;
			search_results.at(i).best.get_transform(tf);
			std::stringstream filename;
			filename << opts["p"].as<std::string>() << "global_calibration" << "_filter_" << ToString((pcl_filter::Filter3d)opts["filter"].as<int>())
					<< "_N_" << i
					<< "_T_" << search_results.at(i).best.to_simple_string()
					<< ".jpg";
			export_image_with_points<pcl::PointXYZI>(list_images_filtred.at(0),list_points_filtred.at(0),camera_model, tf, filename.str());
		}
	}

	// Calibration done her
	clock_t clock_end = clock();

//	for(int i = 0; i < search_results.size(); ++i){
//		std::cout << i << "_0: " << search_results.at(i).at(0).to_simple_string() << std::endl;
//		std::cout << i << "_1: " << search_results.at(i).at(1).to_simple_string() << std::endl;
//		std::cout << i << "_2: " << search_results.at(i).at(2).to_simple_string() << std::endl;
//		std::cout << i << "_3: " << search_results.at(i).at(3).to_simple_string() << std::endl;
//		std::cout << i << "_4: " << search_results.at(i).at(4).to_simple_string() << std::endl;
//		std::cout << i << "_5: " << search_results.at(i).at(5).to_simple_string() << std::endl;
////		for(int j = 0; j < search_results.at(i).best_results.size(); ++j){
////			std::cout << j << ":\tscore:" << search_results.at(i).best_results.at(j).score << std::endl;
////		}
//	}

	timing.push_back(clock());
	std::cout << "\n\n" << "generating output..." << spacer;
	std::stringstream out;
	out << "\n";
	out << time_string(clock_start, clock_end, "Total Time", iterations, search_results.at(0).best_results.size(), search_results.at(search_results.size()-1).best.score, true);
	out << "\n\n\n";
	out << "nr" << "\t" << search_results.at(0).best.to_description_string() << "\tfc" << std::endl;

	for(int i = 0; i < search_results.size(); ++i){
		out
				<< i << "\t"
				<< search_results.at(i).best.to_simple_string() << "\t"
				<< search_results.at(i).get_fc() << "\t" << std::endl;


	}
	timing.push_back(clock());
	std::cout << spacer << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << std::endl;

	//std::cout << out.str();
	if(!filename.empty()){
		myfile << out.str();
		myfile.close();
	}

	return 0;
}
