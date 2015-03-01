#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <kitti_calibration/gui/gui_opencv.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;


int main(int argc, char* argv[]){
	printf("Starting \n");

	std::stringstream available_pcl_filters;
//	std::stringstream available_opencv_filters;

	available_pcl_filters << "use the number to select the pcl filter: \n";
	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
		available_pcl_filters << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n ";
	}

//	available_opencv_filters << "use numbers to select pcl filters: \n";
//	for (int i = 0; i < pcl_filter::NR_ENUMS; ++i){
//		available_opencv_filters << i << ": " << ToString((pcl_filter::Filter3d)i) << "\n ";
//	}

	std::vector<std::string> kitti_files;
	std::vector<bool> enabled_windows(kitti_calibration::window_name::NR_WINDOWS);

	for (int i = 0; i < enabled_windows.size(); ++i){
		enabled_windows[i] = 1;
	}

	po::options_description desc("Usage");
	desc.add_options()
	("help", "Print this help messages")
	("i", po::value<std::vector<std::string> >(&kitti_files)->required()->multitoken(), "kitti dataset to load")
	("camera", po::value<int>()->default_value(0), "camera")
	("seq", po::value<int>()->default_value(0), "sequence number")
	("filter3d", po::value<int>()->default_value(0), available_pcl_filters.str().c_str())
	("create_imagefiles", po::value<bool>()->default_value(false), "save create image file of current view" )
	("gui", po::value< std::vector <bool > >(&enabled_windows)->multitoken(), "GUI elements to show: MAIN,IMAGE,TRANSFORM,CONFIG,CAMERA\nexample  1 1 0 1 1 disables the manual tf gui");
//		("filter2d", po::value<int>(), available_pcl_filters.str().c_str())
	;

	po::variables_map opts;
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

	if(kitti_files.size() == 0){
		std::cout 	<< "Missing kitti data path (--i)" << std::endl
		<< desc << std::endl;
	}

	if ( opts.count("help")  )
	{
		std::cout 	<< "Kitti gui" << std::endl
					<< desc << std::endl;
		return 0;
	}

	kitti_calibration::Gui_opencv gui;
	gui.set_gui(enabled_windows);
	gui.set_export_images(opts["create_imagefiles"].as<bool>());
	gui.init(kitti_files, opts["camera"].as<int>(), opts["seq"].as<int>(),opts["filter3d"].as<int>());

	return 0;
}
