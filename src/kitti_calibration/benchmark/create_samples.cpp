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
	tf::Transform tf_result = (cam0_to_cam * velo_to_cam0*search_startpoint);
	image_cloud::transform_pointcloud(transformed, tf_result);

	//	// Transform Manual
	//	image_cloud::transform_pointcloud(transformed, search_startpoint);

	// Messure time for image preparation
	clock_t time_start = clock();
	for (int l = 0; l < loop; ++l) {
		// create inverse imageimage_pointcloud::
		image_cloud::create_inverse_transformed(image_load, image_inverse);
	}
	clock_t time_end = clock();
	std::ofstream myfile;
	std::stringstream filename_results;
	filename_results << path_out << "results_C_" << camera << "_SEQ_"
			<< sequence << "_P_IN_" << transformed.size() << "_" << loop << ".txt";
	myfile.open(filename_results.str().c_str());
	std::cout
			<< time_string(time_start, time_end, "image_filters", loop,
					image_load.rows, image_load.cols, 0, true);
	myfile
			<< time_string(time_start, time_end, "image_filters", loop,
					image_load.rows, image_load.cols, 0, true);

	for (int i = 0; i < 9; ++i) {
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

		cv::Mat projected_inversed, projected_ori, projected_depth;
		image_inverse.copyTo(projected_inversed);
		image_load.copyTo(projected_ori);
		projected_depth = cv::Mat::zeros(image_inverse.rows, image_inverse.cols, image_inverse.type());
		project2d::project_2d(camera_model, points_filtred, projected_inversed,
				project2d::DEPTH);
		project2d::project_2d(camera_model, points_filtred, projected_ori,
				project2d::DEPTH);
		project2d::project_2d(camera_model, points_filtred, projected_depth,
						project2d::DEPTH);
		long unsigned score = 0;
		score::objective_function<pcl::PointXYZI, uchar>(camera_model,
				points_filtred, image_inverse, score);

		std::cout
				<< time_string(time_start, time_end,
						ToString((pcl_filter::Filter3d) i), loop,
						transformed.size(), points_filtred.size(), score);
		myfile
				<< time_string(time_start, time_end,
						ToString((pcl_filter::Filter3d) i), loop,
						transformed.size(), points_filtred.size(), score);

		std::stringstream filename;
		filename << path_out << "projected_C_" << camera << "_SEQ_" << sequence << "_"
				<< ToString((pcl_filter::Filter3d) i)
				<< "_P_IN_" << transformed.size()
				<< "_P_OUT_"<< points_filtred.size()
				<< "_S_" << score;
		imwrite(filename.str() + "_ori.jpg", image_load);
		imwrite(filename.str() + "_inv.jpg", image_inverse);
		imwrite(filename.str() + "_p_ori.jpg", projected_ori);
		imwrite(filename.str() + "_p_inv.jpg", projected_inversed);
		imwrite(filename.str() + "_p_depth.jpg", projected_depth);
	}
	myfile.close();
}

int main(int argc, char* argv[]){

	std::vector<float> tf;
	tf.resize(6);

	for (int i = 0; i < 6; ++i){
		tf[i] = 0;
	}

	po::options_description desc("Usage");
	desc.add_options()
	("help", "Print this help messages")
	("i", po::value<std::string>()->default_value("/media/Daten/kitti/kitti/2011_09_26_drive_0005_sync/"), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera")
	("s", po::value<int>()->default_value(0), "sequence number")
	("l", po::value<int>()->default_value(1), "number of iterations")
	("w", po::value<bool>()->default_value(true), "write image files")
	("f", po::value<bool>()->default_value(false), "process full kitti dataset")
	("tf", po::value< std::vector <float > >(&tf)->multitoken(), "transforme: x y z roll pitch yaw")
	("p", po::value<std::string>()->default_value(""), "output file prefix")
	("skip", po::value<bool>()->default_value(false), "skip slow algorithms");

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
		std::cout 	<< "Kitti offline edge calibration" << std::endl
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
	}

	std::string data_path = opts["i"].as<std::string>();
	int camera = opts["camera"].as<int>();
	int sequence = opts["s"].as<int>();
	int loop = opts["l"].as<int>();
	bool create_files = opts["w"].as<bool>();
	bool process_all = opts["f"].as<bool>();
	std::string path_out = opts["p"].as<std::string>();

	tf::Transform search_startpoint;
	search_startpoint.setIdentity();
	search_startpoint.setOrigin(tf::Vector3(tf[0],tf[1],tf[2]));
	tf::Quaternion q;
	q.setRPY(tf[3],tf[4],tf[5]);
	search_startpoint.setRotation(q);

	kitti::Dataset data(data_path);

	if(process_all){
		for(sequence = 0; sequence < data.pointcloud_file_list.size(); ++sequence){
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


