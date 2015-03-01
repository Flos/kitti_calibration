/*
 * Simplegui.h
 *
 *  Created on: 17.01.2015
 *      Author: fnolden
 */

#include <iostream>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// ROS
#include <image_geometry/pinhole_camera_model.h>

// Kitti
#include <kitti/common/serialization/camera.h>
#include <kitti/common/serialization/tf.h>
#include <kitti/common/serialization/dataset.h>
#include <kitti/common/serialization/file_list.h>


#include <image_cloud/common/calibration/pipeline/enums.h>

// Own
#include <kitti_calibration/gui/slider.h>

#ifndef SRC_CALIBRATION_SIMPLE_GUI_H_
#define SRC_CALIBRATION_SIMPLE_GUI_H_

namespace kitti_calibration {


namespace image_filter{

	enum image_index{
		FILE_READ = 0,
		IMAGE_GREY = 1,
		IMAGE_BLUR = 2,
		IMAGE_EDGE = 3,
		IMAGE_INVERSE_TRANSFORMED = 4,
		IMAGE_FULL = 5,
		IMAGE_POINTS = 6
	};

	struct Image_filter{
		Slider blur;
		std::vector< std::vector<Slider> >blur_values;
		Slider edge;
		std::vector<std::vector<Slider> >edge_values;
		std::string window_name;
	};

};

struct Set_selector{
	kitti::String_list images;
	kitti::String_list pointclouds;
	Slider pos;
};

namespace window_name{

enum window_name{
	MAIN = 0,
	IMAGE = 1,
	TRANSFORM = 2,
	CONFIG = 3,
	CAMERA = 4,
	NR_WINDOWS = 5
};

}

struct Window_names{
	std::vector<std::string> names;
	std::string at(unsigned int i){
		return names.at(i);
	}

	void resize(unsigned int i){
		names.resize(i);
	}

	std::string& operator[] (const int nIndex)
	{
	    return names[nIndex];
	}
};

struct Dataset_config{
	Slider pos_image;
	Slider pos_camera;
};

//Date for all
struct Datasets_list{
	std::vector<kitti::Dataset> list_datasets;
	std::vector<Dataset_config> list_config;
	std::vector<std::vector<Slider> >filter3d_data;
	image_filter::Image_filter filter2d;
	Slider pos_dataset;
	Slider processed_image_selector;
	Slider projection;
	Slider pcl_filter;
};

class Gui_opencv {

public:
	Gui_opencv();
	virtual ~Gui_opencv();
	void set_gui(std::vector<bool> enabled_windows);
	void init(std::vector<std::string> paths_datas, int camera = 0, int seq = 0, int pcl_filter = pcl_filter::DEPTH_INTENSITY_AND_REMOVE_CLUSER_2D);

	void load_image();
	bool load_pcl();
	void load_projection();

	void update_view();

	void filter3d();
	void filter2d();

	void project2image(cv::Mat &image, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

	void create_gui_general_conf();
	void create_gui_filter2d();
	void create_gui_filter3d();
	void create_static_gui();
	void create_gui_camera();
	void create_gui_manual_tf();

	void recreate_config_gui();

	void update_values();
	void update_image();

	void loop();

	void init_menu_options();
	void init_tf();

	void init_datasets(int camera = 0, int seq = 0);

	Window_names window_names;
	std::vector<bool> windows_enabled;

	Datasets_list datasets;
	std::vector<std::string> config_files;

	std::vector<cv::Mat> images;
	boost::mutex filter_lock;

	// config common
	std::vector<std::string> filter3d_names;
	std::vector<std::string> filter2d_blur_names;
	std::vector<std::string> filter2d_edge_names;
	//std::vector<std::vector<Filter_value> >filter_data;
	Slider tf_data[6];

	Slider camera_slider[4];

	image_geometry::PinholeCameraModel camera_model;
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_file;
};

} /* namespace kitti */

#endif /* SRC_CALIBRATION_SIMPLE_GUI_H_ */
