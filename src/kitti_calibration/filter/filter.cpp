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
#include <kitti/common/serialization/filenames.h>
#include <iostream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

po::variables_map opts;


int main(int argc, char* argv[]){
	std::vector<clock_t> timing;
	timing.push_back(clock());
	std::stringstream available_pcl_filters;

	available_pcl_filters << "pcl filter: \n";
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		available_pcl_filters << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n ";
	}

	std::vector<double> start(6);

	for (int i = 0; i < 6; ++i){
		start[i] = 0;
	}

	po::options_description desc("Usage");
	desc.add_options()
	("help", "Help message")
	("i", po::value<std::string>()->required(), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera")
	("seq", po::value<int>()->default_value(0), "sequence number")
	("o", po::value<std::string>()->required(), "filtred pcl file")
	("tf", po::value< std::vector <double > >(&start)->multitoken(), "tf: pointcloud -> camera x y z roll pitch yaw")
	("tf-inv", po::value< std::vector <double > >(&start)->multitoken(), "tf: camera -> pointcloud: x y z roll pitch yaw")
	("filter", po::value<int>()->default_value(pcl_filter::DEPTH_INTENSITY), available_pcl_filters.str().c_str());

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
		std::cout 	<< "kitti pcl filter" << std::endl
					<< desc << std::endl;
		return 0;
	}


	std::cout << "\nstarting..." << std::endl;


	// init tf from commandline
	search::search_value start_tf(start[0], start[1], start[2], start[3], start[4], start[5], 0);

	if(opts.count("tf-inv")){
		start_tf = search::search_value(start_tf.get_transform().inverse());
	}

	int camera = opts["camera"].as<int>();
	int windows_size = 1;

	kitti::Dataset data;
	data.init(opts["i"].as<std::string>());

	image_geometry::PinholeCameraModel camera_model;

	std::deque<cv::Mat> list_images;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_filtred_points(windows_size);

	load_kitti_data(data, list_images, list_points, camera_model, camera, opts["seq"].as<int>(), 1);

	std::cout << "Filter " << ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << " started...";
	timing.push_back(clock());

	// Transfrom using the manual tf
	image_cloud::transform_pointcloud( list_filtred_points.at(0), start_tf.get_transform());

	image_cloud::filter3d_switch(list_points.at(0), list_filtred_points.at(0),
			camera_model,
			(pcl_filter::Filter3d)opts["filter"].as<int>(),
			list_images.at(0).rows, list_images.at(0).cols);

	timing.push_back(clock());
	std::cout << time_diff(timing[timing.size()-2], timing[timing.size()-1]) << " s\n\n";

	// Points are transformed into the choosen camera frame... Transform them back to the pointcloud origin
	image_cloud::transform_pointcloud( list_filtred_points.at(0), start_tf.get_transform().inverse());

	transform_points_to_origin(data, camera,  list_filtred_points.at(0));

	kitti::filenames::create_folder(opts["o"].as<std::string>());
	kitti::io::save_pointcloud(opts["o"].as<std::string>(),  list_filtred_points.at(0));
	std::cout << "out: " <<  list_filtred_points.at(0).size();

	return 0;
}
