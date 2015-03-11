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
	std::cout << "\nstarting..." << std::endl;

	std::vector<clock_t> timing;
	timing.push_back(clock());
	std::stringstream available_pcl_filters;

	available_pcl_filters << "pcl filter: \n";
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		if(i<10) available_pcl_filters << " ";
		available_pcl_filters << " " << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n";
	}

	std::stringstream available_edge_filters;
	available_edge_filters << "image edge filters: \n";
	for (int i = 0; i < image_filter::edge::NR_ENUMS; ++i){
		if(i<10) available_pcl_filters << " ";
		available_edge_filters << i << ": " << ToString((image_filter::edge::Edge)i) << "\n";
	}
	std::vector<double> range;
	std::vector<double> start;
	std::vector<int> steps;
	range.resize(6);
	start.resize(6);
	steps.resize(6);

	// init weight for score function
	std::vector<bool> use_weight(pcl_filter::NR_ENUMS);
	for(int i=0; i<use_weight.size(); ++i){
		use_weight[i] = 0;
	}

	int iterations = 1;

	char labels[6] = {'x','y','z','r','p','y'};

	for (int i = 0; i < 6; ++i){
		steps[i] = 3;
		range[i] = 0.01;
		start[i] = 0;
	}
	//std::vector<double>() range {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
	po::options_description desc("Usage");
	desc.add_options()
	("help", "Print this help messages")
	("i", po::value<std::string>()->default_value("/media/Daten/kitti/kitti/2011_09_26_drive_0005_sync/"), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera nr")
	("seq", po::value<int>()->default_value(0), "sequence number")
	("window", po::value<int>()->default_value(1), "sliding window size")
	("range", po::value< std::vector<double> >(&range)->multitoken(), "search range: x y z roll pitch yaw")
	("steps", po::value< std::vector<int> >(&steps)->multitoken(), "steps: x y z roll pitch yaw")
	("tf", po::value< std::vector <double > >(&start)->multitoken(), "tf start: pointcloud -> camera: x y z roll pitch yaw")
	("save_images", po::value<bool>()->default_value(true), "write image files")
	("p", po::value<std::string>()->default_value(""), "output file prefix")
	("log",	po::value<std::string>()->default_value("calibration.log"),"calibration log file")
	("blur",	po::value<bool>()->default_value(true),"blur images")
	("weight", po::value<bool>()->default_value(true), "use weighted score")
	("filter", po::value<int>()->default_value(pcl_filter::DEPTH_EDGE_PROJECTION), available_pcl_filters.str().c_str())
	("edge", po::value<int>()->default_value(image_filter::edge::MAX), available_edge_filters.str().c_str());


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

	if ( opts["weight"].as<bool>() ){
		use_weight[pcl_filter::DEPTH_EDGE_PROJECTION] = true;
		use_weight[pcl_filter::DEPTH_RADIUS] = true;
		use_weight[pcl_filter::DEPTH_EDGE_PROJECTION_AGGREGATED] = true;
		use_weight[pcl_filter::DEPTH_INTENSITY_AND_REMOVE_CLUSER_2D] = true;
		use_weight[pcl_filter::REMOVE_CLUSER_2D_RADIUS_SEARCH] = true;
		use_weight[pcl_filter::REMOVE_CLUSTER_2D] = true;
	}

	search::search_value best_result(start[0], start[1], start[2], start[3], start[4], start[5], 1);

	std::cout << "Opening KITTI dataset..." << spacer;
	timing.push_back(clock());

	kitti::Dataset data(opts["i"].as<std::string>());
	timing.push_back(clock());

	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	// Create buffers for search and results
	search::search_setup_vector search_configs;
	search::multi_search_results_vector search_results;

	search::search_value search_chosen_tfs;

	// load and prepare kitti data
	std::deque<cv::Mat> list_images_raw, list_images_raw_window, list_images_window;
	std::deque<pcl::PointCloud<pcl::PointXYZI> > list_points_raw, list_points_raw_window, list_points_window;
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
			data.pointcloud_file_list.size()-opts["seq"].as<int>()
			);

	assert( list_images_raw.size() > 0);
	assert( list_images_raw.size() == list_points_raw.size());
	assert( list_images_raw.size() >= opts["window"].as<int>());


	list_images_raw_window.assign(list_images_raw.begin(), list_images_raw.begin() + opts["window"].as<int>());
	list_points_raw_window.assign(list_points_raw.begin(), list_points_raw.begin() + opts["window"].as<int>());


	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";

	// filter images
	std::cout << "pre filtering images..." << spacer;
	pre_filter_images(list_images_raw_window, list_images_window, opts["blur"].as<bool>(), (image_filter::edge::Edge)opts["edge"].as<int>());

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";


	// Pre filter points
	std::cout << "pre filtering points with filter " << ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << "..." << spacer;

	pre_filter_points(list_points_raw_window, camera_model,
			(pcl_filter::Filter3d)opts["filter"].as<int>(),
			list_points_window,
			list_images_raw.at(0).rows,
			list_images_raw.at(0).cols,
			best_result
			);


	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";
	std::cout << "Total points: " << sum_points(list_points_raw) << " after filter: " << sum_points(list_points_window) << std::endl;


	// Create detailed log file
	std::ofstream file_log;
	std::string filename = opts["p"].as<std::string>()+opts["log"].as<std::string>();
	if(!opts["p"].as<std::string>().empty()){

		kitti::filenames::create_folder(filename.c_str());

		file_log.open(filename.c_str());

		//Push all settings into the config
		file_log << "calib_time:" << spacer << datetime() << std::endl;
		file_log << "cmdline:" << spacer;
		for(int i = 0; i < argc; ++i){
			file_log << std::string(argv[i]) << spacer;
		}
		file_log << std::endl;
		file_log.flush();
	}


	std::cout << "started calibration...\n\n";
	clock_t clock_start = clock();

	unsigned int image_nr = 0;
	long unsigned int total_points = sum_points(list_points_raw);
	long unsigned int total_points_filtred = sum_points(list_points_window);
	// Calibration starts here


	// Loop sequence
	for(int i = 0; i+opts["window"].as<int>() < list_points_raw.size(); ++i){

		timing.push_back(clock());
		std::stringstream out;

		// setup grid search
		search::search_setup search_config(best_result, range, steps);

		search::search_value_vector current_results;
		search::grid_setup(search_config, current_results);

		// Add process next dataset
		if( i != 0 ){
			// Remove oldest elements
			list_images_window.pop_front();
			list_points_window.pop_front();

			// Filter image and pointcloud
			pcl::PointCloud<pcl::PointXYZI> filtred_points;
			cv::Mat filtred_image;

			pre_filter_image(list_images_raw.at(i + opts["window"].as<int>()), filtred_image, opts["blur"].as<bool>(), (image_filter::edge::Edge)opts["edge"].as<int>());
			pre_filter_points(list_points_raw.at(i + opts["window"].as<int>()),
													camera_model,
													(pcl_filter::Filter3d)opts["filter"].as<int>(),
													filtred_points,
													list_images_raw.at(0).rows,
													list_images_raw.at(0).cols,
													best_result );

			total_points_filtred+= filtred_points.size();

			// Add new data to window
			list_images_window.push_back(filtred_image);
			list_points_window.push_back(filtred_points);
		}

		// run grid search
		search::calculate<pcl::PointXYZI,uchar>(camera_model,
												list_points_window,
												list_images_window,
												current_results,
												true,
												use_weight[opts["filter"].as<int>()]);


		// evaluate result
		search::multi_search_results multi_result;
		multi_result.evaluate(current_results);

		best_result = multi_result.best;

		// Create table for console output
		if(i==0){
			// print description
			out << "N" << spacer << best_result.to_description_string() << spacer
					<< "fc" << spacer << "total";
			out << spacer << "time" << spacer;
			out << std::endl;
		}

		out << i << spacer << best_result.to_simple_string() << spacer
					<< multi_result.get_fc() << spacer << multi_result.size();
		// Print previous loop time
		out << spacer << time_diff(timing.at(timing.size()-1), clock());
		out << std::endl;



		// Output
		std::cout << out.str();
		std::cout.flush();

		if(!filename.empty()){
			file_log << out.str();
			file_log.flush();
		}


		// Create images of all best matching
		if(opts["save_images"].as<bool>()){
			tf::Transform tf;
			best_result.get_transform(tf);
			std::stringstream filename;
			filename << opts["p"].as<std::string>() << "calibration_online_" << "_filter_" << ToString((pcl_filter::Filter3d)opts["filter"].as<int>())
					//<< "_R_" << r
					//<< "_N_" << i
					//<< "_T_" << best_result.to_simple_string()
					<< "_"
					<< kitti::filenames::sequence_number(image_nr) // simple numbering
					<< ".jpg";
			++image_nr;
			export_image_with_points<pcl::PointXYZI>(list_images_window.at(0), list_points_window.at(0), camera_model, tf, filename.str());
		}
	}

	// Calibration done her
	clock_t clock_end = clock();

	// Result output
	std::stringstream out;
	out << std::endl;
	out << "Done";
	out << std::endl;
	out << "Processed " << iterations << " data pairs and used "
			<< time_diff(timing.at(0), timing.at(timing.size()-1)) << " cpu seconds" <<  std::endl;
	out << std::endl;
	out << "camera" << spacer
			<< best_result.to_description_string() << spacer
			<< "iterations" << spacer
			<< "sequence" << spacer
			<< "window" << spacer
			<< "time" << spacer
			<< "filter" << spacer
			<< "points_raw" << spacer
			<< "points_filtred" << spacer
			<< std::endl;
	out << opts["camera"].as<int>() << spacer
			<< best_result.to_simple_string() << spacer
			<< iterations << spacer
			<< opts["seq"].as<int>() << spacer
			<< opts["window"].as<int>() << spacer
			<< time_diff(timing.at(0), timing.at(timing.size()-1)) << spacer
			<< ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << spacer
			<< total_points << spacer
			<< total_points_filtred << spacer
			<< std::endl;
	out << std::endl;
	out << "pointcloud -> camera:" << spacer << best_result.to_simple_string() << std::endl;
	out << "camera -> pointcloud:" << spacer << search::search_value(best_result.get_transform().inverse(), best_result.score).to_simple_string() << std::endl;
	out << std::endl;

	file_log << out.str();
	std::cout << out.str();

	if(!filename.empty()){
		//file_log << out.str();
		file_log.close();
	}

	return 0;
}
