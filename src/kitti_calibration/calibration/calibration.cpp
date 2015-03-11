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
	("f", po::value<bool>()->default_value(false), "process full kitti dataset")
	("p", po::value<std::string>()->default_value(""), "output file prefix")
	("iterations", po::value<int>(&iterations), "optimization iteration (halves search range per iteration")
	("log",	po::value<std::string>()->default_value("calibration.log"),"calibration log file")
	("blur",	po::value<bool>()->default_value(true),"blur images")
	("factor",	po::value<double>()->default_value(0.5),"reduce range step size per iteration using this factor")
	("precision", po::value<double>(), "if set, calibration runs until precision is reached")
	("weight", po::value<bool>()->default_value(true), "use weighted score")
	("pre_filter", po::value<bool>()->default_value(true), "apply point filter once, before search transformations")
	("restarts", po::value<int>()->default_value(1), "number of restarts at final destination")
	("restart_filter", po::value<bool>()->default_value(true), "filter again after restart of calibration")
	("min", po::value<bool>()->default_value(true), "start with minimal step size")
	("filter", po::value<int>()->default_value(pcl_filter::DEPTH_INTENSITY), available_pcl_filters.str().c_str())
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

	if( opts.count("precision") )
	{
		if(!(opts["factor"].as<double>() < 1)){
			std::cout << "ERROR precision cannot be reached. Factor needs to be below of 1.0 but set to: " << opts["factor"].as<double>() << std::endl;
			return 1;
		}

		iterations = 1;
		double max_translation = 0;
		double max_rotation = 0;

		for(int i = 0; i < 3; ++i){
			max_translation = MAX(max_translation, range[i]);
			max_rotation = MAX(max_rotation, range[3+i]);
		}

		while(max_translation > opts["precision"].as<double>()
				|| max_rotation > opts["precision"].as<double>() ){

			max_translation = max_translation*opts["factor"].as<double>();
			max_rotation = max_rotation*opts["factor"].as<double>();

			++iterations;
		}
		std::cout 	<< "\nInfo: Precision was set to: " << opts["precision"].as<double>() << ", set iterations to: " << iterations << std::endl;
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
	std::cout << "pre filtering images..." << spacer;
	pre_filter_images(list_images_raw, list_images_filtred, opts["blur"].as<bool>(), (image_filter::edge::Edge)opts["edge"].as<int>());

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";


	// Pre filter points
	std::cout << "pre filtering points with filter " << ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << "..." << spacer;
	if(opts["pre_filter"].as<bool>()){
		pre_filter_points(list_points_raw, camera_model,
				(pcl_filter::Filter3d)opts["filter"].as<int>(),
				list_points_filtred,
				list_images_raw.at(0).rows,
				list_images_raw.at(0).cols,
				best_result
				);
	}
	else{
		list_points_filtred = list_points_raw;
	}

	timing.push_back(clock());
	std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";
	std::cout << "Total points: " << sum_points(list_points_raw) << " after filter: " << sum_points(list_points_filtred) << std::endl;


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
	search::search_value best_result_before_restart = best_result;
	clock_t clock_start = clock();

	unsigned int image_nr = 0;
	// Calibration starts here
	for(int r = 0; r < opts["restarts"].as<int>(); ++r)
	{
		if(r != 0)
		{
			// Done?
			if(best_result_before_restart == best_result)
			{
				std::stringstream out;
				out << std::endl;
				out << "Done";
				out << std::endl;
				out << "Maximized score after " << r * iterations << " iterations and "
						<< time_diff(timing.at(0), timing.at(timing.size()-1)) << " cpu seconds" <<  std::endl;
				out << std::endl;
				out << "camera" << spacer
						<< best_result.to_description_string() << spacer
						<< "precision" << spacer
						<< "iterations" << spacer
						<< "sequence" << spacer
						<< "window" << spacer
						<< "time" << spacer
						<< "filter" << spacer
						<< "points_filtred" << spacer
						<< "points_raw" << spacer
						<< std::endl;
				out << opts["camera"].as<int>() << spacer
						<< best_result.to_simple_string() << spacer
						<< opts["precision"].as<double>() << spacer
						<< r*iterations << spacer
						<< opts["seq"].as<int>() << spacer
						<< opts["window"].as<int>() << spacer
						<< time_diff(timing.at(0), timing.at(timing.size()-1)) << spacer
						<< ToString((pcl_filter::Filter3d)opts["filter"].as<int>()) << spacer
						<< sum_points(list_points_filtred) << spacer
						<< sum_points(list_points_raw)
						<< std::endl;
				out << std::endl;
				out << "pointcloud -> camera:" << spacer << best_result.to_simple_string() << std::endl;
				out << "camera -> pointcloud:" << spacer << search::search_value(best_result.get_transform().inverse(), best_result.score).to_simple_string() << std::endl;
				out << std::endl;

				file_log << out.str();
				std::cout << out.str();
				break;
			}

			best_result_before_restart = best_result;
			std::cout << "\n\nRestarting...\n";

			// Filter again?
			if(opts["restart_filter"].as<bool>()){
				timing.push_back(clock());
				std::cout << "filter pointclouds... ";
				list_points_filtred.clear();

				pre_filter_points(list_points_raw, camera_model,
						(pcl_filter::Filter3d)opts["filter"].as<int>(),
						list_points_filtred,
						list_images_raw.at(0).rows,
						list_images_raw.at(0).cols,
						best_result
						);
				timing.push_back(clock());
				std::cout << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << "\n";
				std::cout << "Total points: " << sum_points(list_points_raw) << " after filter: " << sum_points(list_points_filtred) << std::endl;

			}
		}

		// Reset range values
		std::vector<double> temp_range;
		temp_range.resize(6);
		for(int j = 0; j<6 ;++j){
			if(opts["min"].as<bool>())
			{
				temp_range[j] = range[j]*pow(opts["factor"].as<double>(),(iterations-1));
			}
			else{
				temp_range[j] = range[j]*pow(opts["factor"].as<double>(),(iterations-1));
			}
		}
		search::search_value previous_result = best_result;

		// Loop scale factor
		for(int i = 0; i < iterations; ++i){
			timing.push_back(clock());
			std::stringstream out;

			// setup grid search
			search_results.get_best_search_value(best_result);


			search::search_setup search_config(best_result, temp_range, steps);

			previous_result = best_result;


			search::search_value_vector current_results;
			search::grid_setup(search_config, current_results);


			// store current config
			search_configs.push_back(search_config);


			// run grid search
			search::calculate<pcl::PointXYZI,uchar>(camera_model,
													list_points_filtred,
													list_images_filtred,
													current_results,
													opts["pre_filter"].as<bool>(),
													use_weight[opts["filter"].as<int>()]);


			// evaluate result
			search::multi_search_results multi_result;
			multi_result.evaluate(current_results);


			// store results
			search_results.push_back(multi_result);
			search_results.get_best_search_value(best_result);


			// Create table for console output
			if(i==0){
				if(r==0)
				{
					out << "\n" << "Expected maximum calibration time: ";
					time_t max_time = clock() - timing.at(timing.size()-1);
					max_time = (max_time * opts["restarts"].as<int>() * iterations)/(60*omp_get_num_threads());
					out << time_diff(timing.at(timing.size()-1), max_time) << " minutes";
					out << "\n\n";
				}
				// print description
				search::search_value search_description;
				out << "R" << spacer << "N" << spacer << search_description.to_description_string() << spacer
						<< "fc" << spacer << "total";
				for(int j = 0; j<6; ++j){
					out << spacer << labels[j];
				}
				out << spacer << "time" << spacer;
				out << std::endl;
			}
			else{
				// Print previous search result
				out <<  r << spacer << i-1 << spacer
									<< best_result.to_simple_string() << spacer
									<< search_results.at(i-1).get_fc() << spacer
									<< search_results.at(i-1).best_results.size();

				// print old and calulate new range
				for(int j = 0; j<6 ;++j){
					out << spacer << temp_range[j];
					if(opts["min"].as<bool>())
					{
						temp_range[j] = range[j]*pow(opts["factor"].as<double>(),(iterations-(i+1)));
					}
					else{
						temp_range[j] = range[j]*pow(opts["factor"].as<double>(),(i));
					}
				}

				// Print previous loop time
				out << spacer << time_diff(timing.at(timing.size()-1), clock());
				out << std::endl;
			}


			// Output
			std::cout << out.str();
			std::cout.flush();

			if(!filename.empty()){
				file_log << out.str();
				file_log.flush();
			}


			// Create images of all best matching
			if(opts["save_images"].as<bool>()
				&&(
					(r == 0 && i == 0)
					|| previous_result.score != best_result.score)
				){
				tf::Transform tf;
				best_result.get_transform(tf);
				std::stringstream filename;
				filename << opts["p"].as<std::string>() << "calibration" << "_filter_" << ToString((pcl_filter::Filter3d)opts["filter"].as<int>())
						//<< "_R_" << r
						//<< "_N_" << i
						//<< "_T_" << best_result.to_simple_string()
						<< "_"
						<< kitti::filenames::sequence_number(image_nr) // simple numbering
						<< ".jpg";
				++image_nr;
				export_image_with_points<pcl::PointXYZI>(list_images_filtred.at(0), list_points_filtred.at(0), camera_model, tf, filename.str());
			}
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

//	timing.push_back(clock());
//	std::cout << "\n\n" << "generating output..." << spacer;
//	std::stringstream out;
//	out << "\n";
//	out << time_string(clock_start, clock_end, "Total Time", iterations, search_results.at(0).best_results.size(), search_results.at(search_results.size()-1).best.score, true);
//	out << "\n\n\n";
//	out << "nr" << "\t" << search_results.at(0).best.to_description_string() << "\tfc" << std::endl;

//	for(int i = 0; i < search_results.size(); ++i){
//		out
//				<< i << "\t"
//				<< search_results.at(i).best.to_simple_string() << "\t"
//				<< search_results.at(i).get_fc() << "\t" << std::endl;
//	}
//
//	timing.push_back(clock());
//	std::cout << spacer << time_diff(timing.at(timing.size()-2), timing.at(timing.size()-1)) << std::endl;

	//std::cout << out.str();
	if(!filename.empty()){
		//file_log << out.str();
		file_log.close();
	}

	return 0;
}
