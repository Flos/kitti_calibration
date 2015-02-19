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

int main(int argc, char* argv[]){

	po::options_description desc("Usage");
	desc.add_options()
	("i", po::value<std::string>()->default_value("/media/Daten/kitti/kitti/2011_09_26_drive_0005_sync/"), "kitti dataset to load");
	desc.add_options()
	("o", po::value<std::string>()->default_value("pre_filtred.txt"), "out file");
	desc.add_options()
	("c", po::value<int>()->default_value(0), "camera");
	desc.add_options()
	("s", po::value<int>()->default_value(0), "sequence number");
	desc.add_options()
	("l", po::value<int>()->default_value(729), "number of iterations");
	desc.add_options()
	("w", po::value<bool>()->default_value(true), "write image files");

	po::variables_map opts;
	po::store(po::parse_command_line(argc, argv, desc), opts);

	try {
	po::notify(opts);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}

	std::string data_path = opts["i"].as<std::string>();
	std::string output_file = opts["o"].as<std::string>();
	int camera = opts["c"].as<int>();
	int sequence = opts["s"].as<int>();
	int loop = opts["l"].as<int>();
	bool create_files = opts["w"].as<bool>();

	tf::Transform search_startpoint, bad_movement;
	bad_movement.setIdentity();
	search_startpoint.setIdentity();

	kitti::Dataset data(data_path);

	image_geometry::PinholeCameraModel camera_model;
	data.camera_list.at(camera).get_camera_model(camera_model);

	cv::Mat image_load, image_inverse;
	data.camera_file_list.at(camera).load_image(image_load, sequence);

	// Process Pointcloud
 	pcl::PointCloud<pcl::PointXYZI> transformed;
 	data.pointcloud_file_list.load_pointcloud(transformed, sequence);
//
//		// Transforms velo_cam0
	tf::Transform velo_to_cam0;
	data.velodyne_to_cam0.get_transform(velo_to_cam0);

	// Transform cam0_to_cam
	tf::Transform cam0_to_cam;

	data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);

	tf::Transform tf_result = (cam0_to_cam*velo_to_cam0);
	image_cloud::transform_pointcloud(transformed, tf_result);

//	// Transform Manual
//	image_cloud::transform_pointcloud(transformed, search_startpoint);

	// Messure time for image preparation
	clock_t time_start = clock();
	for (int l=0; l < loop; ++l){
		// create inverse imageimage_pointcloud::
		image_cloud::create_inverse_transformed(image_load, image_inverse);
	}
	clock_t time_end = clock();


	std::ofstream myfile;
	myfile.open("sample_run.txt");

	std::cout << time_string(time_start, time_end, "image_filters", loop, image_load.rows, image_load.cols, 0, true);
	myfile << time_string(time_start, time_end, "image_filters", loop, image_load.rows, image_load.cols, 0, true);

	for(int i=0; i<9; ++i){
		time_start = clock();
		for(int l=0; l< loop; ++l){
			// Project filter
			pcl::PointCloud<pcl::PointXYZI> points_filtred;
			image_cloud::filter3d_switch<pcl::PointXYZI>(transformed, points_filtred, camera_model, camera, sequence, (pcl_filter::Filter3d)i, image_load.rows, image_load.cols);

		}
		time_end = clock();

		// Create image output
		pcl::PointCloud<pcl::PointXYZI> points_filtred;
		assert(points_filtred.size() == 0);
		image_cloud::filter3d_switch<pcl::PointXYZI>(transformed, points_filtred, camera_model, camera, sequence, (pcl_filter::Filter3d)i, image_load.rows, image_load.cols);


		cv::Mat projected_inversed, projected_ori;
		image_inverse.copyTo(projected_inversed);
		image_load.copyTo(projected_ori);
		project2d::project_2d(camera_model, points_filtred, projected_inversed, project2d::DEPTH);
		project2d::project_2d(camera_model, points_filtred, projected_ori, project2d::DEPTH);
		long unsigned score = 0;
		score::objective_function<pcl::PointXYZI,uchar>(camera_model, points_filtred, image_inverse, score);

		std::cout << time_string(time_start, time_end, ToString((pcl_filter::Filter3d)i), loop, transformed.size(), points_filtred.size(), score);
		myfile << time_string(time_start, time_end, ToString((pcl_filter::Filter3d)i), loop, transformed.size(), points_filtred.size(), score);

		std::stringstream filename;
		filename << "projected_C_" << camera << "_SEQ_" << sequence << "_P_IN_" << transformed.size() << "_" << ToString((pcl_filter::Filter3d)i) << "_P_OUT_" << points_filtred.size() << "_S_"<< score;
		imwrite(filename.str()+"_ori.jpg", projected_ori);
		imwrite(filename.str()+"_inv.jpg", projected_inversed);
	}
	myfile.close();

//	//

	return 0;
}


