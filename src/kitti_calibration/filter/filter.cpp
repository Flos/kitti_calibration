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

	po::options_description desc("Usage");
	desc.add_options()
	("help", "Help message")
	("i", po::value<std::string>()->required(), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera")
	("seq", po::value<int>()->default_value(0), "sequence number")
	("o", po::value<std::string>()->required(), "filtred pcl file")
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

	image_geometry::PinholeCameraModel camera_model;
	pcl::PointCloud<pcl::PointXYZI> in_points, out_points;
	cv::Mat image;

	kitti::Dataset data;
	data.init(opts["i"].as<std::string>());
	data.camera_list.at(opts["camera"].as<int>()).get_camera_model(camera_model);
	data.pointcloud_file_list.load_pointcloud(in_points, opts["seq"].as<int>());
	data.camera_file_list.at(opts["camera"].as<int>()).load_image(image, opts["seq"].as<int>());


	std::cout << "Filter " << ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << " started...";
	timing.push_back(clock());

	image_cloud::filter3d_switch(in_points, out_points, camera_model, (pcl_filter::Filter3d)opts["filter"].as<int>(), image.rows, image.cols);

	timing.push_back(clock());
	std::cout << time_diff(timing[timing.size()-2], timing[timing.size()-1]) << " s\n\n";

	kitti::filenames::create_folder(opts["o"].as<std::string>());
	kitti::io::save_pointcloud(opts["o"].as<std::string>(), out_points);

	return 0;
}
