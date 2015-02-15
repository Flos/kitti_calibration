#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <kitti_calibration/gui/gui_opencv.h>


int main(){
	printf("Starting \n");
	kitti_calibration::Gui_opencv gui;
	gui.init();

	return 0;
}
