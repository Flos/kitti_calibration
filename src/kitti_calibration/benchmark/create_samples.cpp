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

#include <kitti/common/serialization/filenames.h>
#include <kitti_calibration/common/common.hpp>
#include <kitti_calibration/common/messure_time.h>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

void run_filter_set(kitti::Dataset data, int camera, int sequence, tf::Transform search_startpoint, int loop,
		const std::string& path_out, const bool *enabled) {

	image_geometry::PinholeCameraModel camera_model;
	data.camera_list.at(camera).get_camera_model(camera_model);
	cv::Mat image_load, image_inverse;
	data.camera_file_list.at(camera).load_image(image_load, sequence);

	// Process Pointcloud
	pcl::PointCloud < pcl::PointXYZI > transformed;
	data.pointcloud_file_list.load_pointcloud(transformed, sequence);
	//
	//		// Transforms velo_cam0
	tf::Transform velo_to_cam0;
	data.velodyne_to_cam0.get_transform(velo_to_cam0);
	// Transform cam0_to_cam
	tf::Transform cam0_to_cam;
	data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);
	tf::Transform tf_result = (cam0_to_cam * velo_to_cam0 * search_startpoint);
	image_cloud::transform_pointcloud(transformed, tf_result);

	// Messure time for image preparation
	clock_t time_start = clock();
	for (int l = 0; l < loop; ++l) {
		// create inverse imageimage_pointcloud::
		image_cloud::create_inverse_transformed(image_load, image_inverse);
	}
	clock_t time_end = clock();


	std::ofstream file_log;
	std::stringstream filename_results;
	filename_results << path_out << "results_C_" << camera << "_SEQ_"
			<< sequence << "_P_IN_" << transformed.size() << "_" << loop+1 << ".log";

	kitti::filenames::create_folder(filename_results.str());
	file_log.open(filename_results.str().c_str());

	std::cout
			<< "Nr\t" << time_string(time_start, time_end, "image_filters", loop+1,
					image_load.rows, image_load.cols, 0, true);
	file_log
			<< time_string(time_start, time_end, "image_filters", loop+1,
					image_load.rows, image_load.cols, 0, true);

	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i) {
		std::stringstream out;

		if(!enabled[i]) continue;
		time_start = clock();
		for (int l = 0; l < loop; ++l) {
			// Project filter
			pcl::PointCloud < pcl::PointXYZI > points_filtred;
			image_cloud::filter3d_switch < pcl::PointXYZI> (transformed, points_filtred, camera_model, (pcl_filter::Filter3d) i, image_load.rows, image_load.cols);

		}
		time_end = clock();

		// Create image output
		pcl::PointCloud < pcl::PointXYZI > points_filtred;
		assert(points_filtred.size() == 0);
		image_cloud::filter3d_switch < pcl::PointXYZI> (transformed, points_filtred, camera_model, (pcl_filter::Filter3d) i, image_load.rows, image_load.cols);

		long unsigned score = 0;
		score::objective_function<pcl::PointXYZI, uchar>(camera_model,
				points_filtred, image_inverse, score);

		out << i << "\t"
				<< time_string(time_start, time_end,
						ToString((pcl_filter::Filter3d) i), loop,
						transformed.size(), points_filtred.size(), score);

		file_log << out.str();
		file_log.flush();

		std::cout << out.str();

		std::stringstream filename;
		filename << path_out << "projected_C_" << camera << "/"
				<< ToString((pcl_filter::Filter3d) i) << "_";

		cv::Mat image_empty;
		image_empty = cv::Mat::zeros(image_inverse.rows, image_inverse.cols, image_inverse.type());

		tf::Transform tf;
		tf.setIdentity();

		pcl::PointCloud < pcl::PointXYZI > empty;
		export_image_with_points(image_load, empty, camera_model, tf, filename.str()+"ori_" + kitti::filenames::sequence_number(sequence) + ".jpg" );
		export_image_with_points(image_load, points_filtred, camera_model, tf, filename.str()+"p_ori_" + kitti::filenames::sequence_number(sequence) + ".jpg" );
		export_image_with_points(image_inverse, points_filtred, camera_model, tf, filename.str()+"inv_" + kitti::filenames::sequence_number(sequence) + ".jpg" );
		export_image_with_points(image_inverse, points_filtred, camera_model, tf, filename.str()+"p_inv_" + kitti::filenames::sequence_number(sequence) + ".jpg" );
		export_image_with_points(image_empty, points_filtred, camera_model, tf, filename.str()+"p_depth_" + kitti::filenames::sequence_number(sequence) + ".jpg" );
		export_image_with_points(image_empty, points_filtred, camera_model, tf, filename.str()+"p_intensity_" + kitti::filenames::sequence_number(sequence) + ".jpg", project2d::INTENSITY);

	}
	file_log.close();
}

int main(int argc, char* argv[]){

	std::stringstream available_pcl_filters;

	available_pcl_filters << "use numbers to select pcl filters: \n";
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		available_pcl_filters << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n ";
	}

	std::vector<double> start;	start.resize(6);
	std::vector<int> enabled_filter;

	for (int i = 0; i < 6; ++i){
		start[i] = 0;
	}

	po::options_description desc("Usage");
	desc.add_options()
	("help", "Print this help messages")
	("i", po::value<std::string>()->default_value("/media/Daten/kitti/kitti/2011_09_26_drive_0005_sync/"), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera")
	("seq", po::value<int>()->default_value(0), "sequence number")
	("l", po::value<int>()->default_value(1), "number of iterations")
	("w", po::value<bool>()->default_value(true), "write image files")
	("run", po::value<bool>()->default_value(false), "process all cameras until dataset ends")
	("tf", po::value< std::vector <double > >(&start)->multitoken(), "tf start: pointcloud -> camera: x y z roll pitch yaw")
	("tf-inv", po::value< std::vector <double > >(&start)->multitoken(), "tf: camera -> pointcloud: x y z roll pitch yaw")
	("prefix", po::value<std::string>()->default_value(""), "output file prefix")
	("skip", po::value<bool>()->default_value(false), "skip slow algorithms")
	("filters", po::value<std::vector <int> >(&enabled_filter)->multitoken(), available_pcl_filters.str().c_str());

	po::variables_map opts;
	//po::store(po::parse_command_line(argc, argv, desc), opts);
	po::store(parse_command_line(argc, argv, desc, po::command_line_style::unix_style ^ po::command_line_style::allow_short), opts);

	try {
		po::notify(opts);
	} catch (std::exception& e) {
		std::cout << "Error: " << e.what() << "\n";
		std::cout << desc << std::endl;
			return 1;
	}

	/** --help option
	 */

	if ( opts.count("help")  )
	{
		std::cout 	<< "Kitti offline image sample creator" << std::endl
					<< desc << std::endl;
		return 0;
	}

	bool enabled[pcl_filter::NR_ENUMS];
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		enabled[i] = true;
	}

	if(opts["skip"].as<bool>()){
		// mark slow algorithms to be skipped
		enabled[pcl_filter::DEPTH_RADIUS] = false;
		enabled[pcl_filter::NORMAL_DIFF] = false;
		enabled[pcl_filter::REMOVE_CLUSER_2D_RADIUS_SEARCH] = false;
		enabled[pcl_filter::REMOVE_CLUSTER_2D] = false;
	}

	if ( opts.count("filters") && !enabled_filter.empty() ){
		// Disable all
		for(int i = 0; i < pcl_filter::NR_ENUMS; ++i)
		{
			enabled[i] = false;
		}

		// Enable selected
		for(int i = 0; i < enabled_filter.size(); ++i){
			enabled[enabled_filter[i]] = true;
		}
	}

	std::string data_path = opts["i"].as<std::string>();
	int camera = opts["camera"].as<int>();
	int sequence = opts["seq"].as<int>();
	int loop = opts["l"].as<int>();
	bool create_files = opts["w"].as<bool>();
	bool process_all = opts["run"].as<bool>();
	std::string path_out = opts["prefix"].as<std::string>();

	// init tf from commandline
	search::search_value start_tf(start[0], start[1], start[2], start[3], start[4], start[5], 0);

	if(opts.count("tf-inv")){
		start_tf = search::search_value(start_tf.get_transform().inverse());
	}

	tf::Transform search_startpoint = start_tf.get_transform();

	kitti::Dataset data(data_path);

	if(process_all){
		for( ; sequence < data.pointcloud_file_list.size(); ++sequence){
			for(camera = 0; camera < data.camera_list.size(); ++camera){
				run_filter_set(data, camera, sequence, search_startpoint, loop, path_out, enabled);
			}
		}
	}
	else{
		run_filter_set(data, camera, sequence, search_startpoint, loop, path_out, enabled);
	}

//	//

	return 0;
}


