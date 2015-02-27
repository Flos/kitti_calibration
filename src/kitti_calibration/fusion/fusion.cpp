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
	("tf", po::value< std::vector <double > >(&start)->multitoken(), "start transforme: x y z roll pitch yaw");

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
	pcl::PointCloud<pcl::PointXYZI> in_points;
	pcl::PointCloud<pcl::PointXYZRGB> out_points;
	cv::Mat image;
	int camera = opts["camera"].as<int>();
	search::search_value start_tf(start[0], start[1], start[2], start[3], start[4], start[5], 0);

	kitti::Dataset data;
	data.init(opts["i"].as<std::string>());
	data.camera_list.at(camera).get_camera_model(camera_model);
	data.pointcloud_file_list.load_pointcloud(in_points, opts["seq"].as<int>());
	data.camera_file_list.at(camera).load_image(image, opts["seq"].as<int>());


	timing.push_back(clock());

	//Transform pointcloud to selected camera
	tf::Transform velo_to_cam0;
	data.velodyne_to_cam0.get_transform(velo_to_cam0);

	// Transform cam0_to_cam
	tf::Transform cam0_to_cam;
	data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);

	tf::Transform result = cam0_to_cam * velo_to_cam0 * start_tf.get_transform();

	image_cloud::transform_pointcloud(in_points, result);

	image_cloud::pointcloud_rgb<pcl::PointXYZI,cv::Vec3b>( camera_model, in_points, image, out_points, opts["color"].as<float>(), opts["distance"].as<float>());

	timing.push_back(clock());
	std::cout << time_diff(timing[timing.size()-2], timing[timing.size()-1]) << " s\n\n";

	kitti::filenames::create_folder(opts["o"].as<std::string>());
	kitti::io::save_pointcloud(opts["o"].as<std::string>(), out_points);

	return 0;
}
