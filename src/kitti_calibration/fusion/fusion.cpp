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
#include <image_cloud/common/pointcloud_rgb.hpp>

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
	("distance", po::value<float>()->default_value(1), "min distance in m")
	("color", po::value<float>()->default_value(0), "min color value")
	("o", po::value<std::string>()->required(), "colored pcl file")
	("tf", po::value< std::vector <double > >(&start)->multitoken(), "tf: pointcloud -> camera: x y z roll pitch yaw");

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

	std::cout << "\nstarting fusion..." << std::endl;

	image_geometry::PinholeCameraModel camera_model;
	pcl::PointCloud<pcl::PointXYZRGB> out_points_rgb;

	int camera = opts["camera"].as<int>();
	int seq = opts["seq"].as<int>();
	int windows_size = 1;

	// init tf from commandline
	search::search_value start_tf(start[0], start[1], start[2], start[3], start[4], start[5], 0);

	kitti::Dataset data;
	data.init(opts["i"].as<std::string>());

	timing.push_back(clock());
	std::deque<cv::Mat> list_images;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_loaded;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_transformed(windows_size);
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_z(windows_size);


	// Pointcloud and Images and Transform pointcloud to selected camera
	load_kitti_data(data, list_images, list_points_loaded, camera_model, camera, seq, windows_size);

	// Transform pointcloud to user tf
	image_cloud::transform_pointcloud<pcl::PointXYZI>(list_points_loaded.at(0), list_points_transformed.at(0), start_tf.get_transform());


	// Filter pointcloud in Z axis (color only first hit from image viewport)
	project2d::Points2d<pcl::PointXYZI> point_map;
	cv::Mat depth_map = cv::Mat::zeros(list_images.at(0).rows, list_images.at(0).cols, CV_8U);
	point_map.init(camera_model, list_points_transformed.at(0), depth_map, project2d::DEPTH);
	point_map.get_points<uchar>(depth_map, list_points_z.at(0), 0);

	// Debug
	{
		std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_edged(windows_size);

		image_cloud::filter3d_switch(list_points_transformed.at(0), list_points_edged.at(0),
				camera_model, pcl_filter::EDGE_IMAGE_PLANE_2D_RADIUS_SEARCH,
				list_images.at(0).rows, list_images.at(0).cols);

	//
		cv::imwrite("fusion_test.jpg", depth_map);

		// Filter edged points in z and project to image
		project2d::Points2d<pcl::PointXYZI> point_map_filtred;
		cv::Mat depth_map_filtred = cv::Mat::zeros(list_images.at(0).rows, list_images.at(0).cols, CV_8U);
		point_map_filtred.init(camera_model, list_points_edged.at(0), depth_map_filtred, project2d::DEPTH);
		cv::imwrite("fusion_test_filtred.jpg", depth_map_filtred);

		cv::Mat image_projected;
		list_images.at(0).copyTo(image_projected);

		project2d::project_2d<pcl::PointXYZI>(camera_model, list_points_edged.at(0),image_projected,project2d::DEPTH);

		cv::imwrite("fusion_test_projected.jpg", image_projected);
	}

	image_cloud::pointcloud_rgb<pcl::PointXYZI, cv::Vec3b>( camera_model, list_points_z.at(0), list_images.at(0), out_points_rgb, opts["color"].as<float>(), opts["distance"].as<float>());

	//Transform pointcloud back

	timing.push_back(clock());
	std::cout << time_diff(timing[timing.size()-2], timing[timing.size()-1]) << " s\n\n";

	kitti::filenames::create_folder(opts["o"].as<std::string>());
	kitti::io::save_pointcloud(opts["o"].as<std::string>(), out_points_rgb);

	return 0;
}
