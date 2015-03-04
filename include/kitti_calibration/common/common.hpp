/*
 * common.hpp
 *
 *  Created on: 23.02.2015
 *      Author: fnolden
 */

#include <image_cloud/common/filter/cv/inverse_distance_transform.hpp>
#include <image_cloud/common/filter/cv/edge.hpp>
#include <image_cloud/common/transform.hpp>
#include <image_cloud/common/calibration/score.hpp>
#include <image_cloud/common/calibration/structs.hpp>
#include <image_cloud/common/calibration/grid_search.hpp>
#include <image_cloud/common/calibration/pipeline/image.hpp>
#include <image_cloud/common/calibration/pipeline/pointcloud.hpp>
#include <image_cloud/common/calibration/pipeline/enums.h>

#include <kitti/common/serialization/filenames.h>

#ifndef INCLUDE_KITTI_CALIBRATION_COMMON_COMMON_HPP_
#define INCLUDE_KITTI_CALIBRATION_COMMON_COMMON_HPP_

template <typename PointT>
void transform_points_to_origin(kitti::Dataset &data, int camera,
		pcl::PointCloud<PointT>& points) {

	// look up transforms
	tf::Transform velo_to_cam0, cam0_to_cam;
	data.velodyne_to_cam0.get_transform(velo_to_cam0);
	data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);

	// Transform points in reverse order using the inverse tf
	image_cloud::transform_pointcloud<PointT>(points, cam0_to_cam.inverse());
	image_cloud::transform_pointcloud<PointT>(points, velo_to_cam0.inverse());
}

template <typename PointT>
void transform_points_to_camera(kitti::Dataset &data, int camera,
		pcl::PointCloud<PointT>& points) {

	// look up transforms
	tf::Transform velo_to_cam0, cam0_to_cam;
	data.velodyne_to_cam0.get_transform(velo_to_cam0);
	data.camera_list.cameras.at(camera).tf_rect.get_transform(cam0_to_cam);

	// Transform points
	image_cloud::transform_pointcloud<PointT>(points, velo_to_cam0);
	image_cloud::transform_pointcloud<PointT>(points, cam0_to_cam);
}

// Pointcloud and Images and Transform pointcloud to selected camera
void load_kitti_data(kitti::Dataset &data,
		std::deque<cv::Mat> &list_images,
		std::deque<pcl::PointCloud<pcl::PointXYZI> > &list_points,
		image_geometry::PinholeCameraModel &camera_model,
		int camera = 0,
		int sequence = 0,
		int window_size = 1
		)
{
	data.camera_list.at(camera).get_camera_model(camera_model);

	for(int i = 0; i < window_size; ++i){
		cv::Mat image_load;
		data.camera_file_list.at(camera).load_image(image_load, sequence+i);

		// create inverse imageimage_pointcloud::
		list_images.push_back(image_load);
		// Image processing done

		// Load Pointcloud
		pcl::PointCloud < pcl::PointXYZI > transformed;
		data.pointcloud_file_list.load_pointcloud(transformed, sequence+i);
		//
		//	Transform PointCloud to camera position
		transform_points_to_camera(data, camera, transformed);
		list_points.push_back(transformed);
	}
}

void pre_filter_images(const	std::deque<cv::Mat> &in_list_images,
								std::deque<cv::Mat> &out_list_images,
								bool blur = true)
{
	out_list_images.resize(in_list_images.size());

	#pragma omp parallel for
	for(int i = 0; i < in_list_images.size(); ++i)
	{
		if(blur)
		{
			cv::Mat blurred;
			cv::GaussianBlur( in_list_images.at(i), blurred, cv::Size(3, 3 ), 0, 0 );
			image_cloud::create_inverse_transformed(blurred, out_list_images.at(i));
		}
		else{
			image_cloud::create_inverse_transformed(in_list_images.at(i), out_list_images.at(i));
		}
	}
}


void pre_filter_points(	const	std::deque<pcl::PointCloud<pcl::PointXYZI> > &in_list_points,
		const image_geometry::PinholeCameraModel &camera_model,
		const pcl_filter::Filter3d filter,
		std::deque<pcl::PointCloud<pcl::PointXYZI> > &out_list_points,
		int rows = 0,
		int cols = 0)
{
	out_list_points.resize(in_list_points.size());

	#pragma omp parallel for
	for(int i = 0; i < in_list_points.size(); ++i)
	{
		image_cloud::filter3d_switch <pcl::PointXYZI > (in_list_points.at(i), out_list_points.at(i), camera_model, filter, rows, cols);
	}
}

template <typename PointT>
void export_image_with_points(const cv::Mat &image,
		const pcl::PointCloud<PointT> &points,
		const image_geometry::PinholeCameraModel &camera_model,
		const tf::Transform &tf,
		std::string filename,
		project2d::Field field = project2d::DEPTH)
{
	cv::Mat projected;
	pcl::PointCloud<PointT> transformed_points = points;

	image.copyTo(projected);

	image_cloud::transform_pointcloud(transformed_points, tf);

	project2d::project_2d<PointT>(camera_model, transformed_points, projected, field);

	kitti::filenames::create_folder(filename);
	imwrite(filename, projected);
}


template <typename PointT>
long unsigned int sum_points(const	std::deque<pcl::PointCloud<PointT> > &in_list_points){
	long unsigned int count = 0;
	for(int i=0; i < in_list_points.size(); ++i){
		count += in_list_points.at(i).size();
	}

	return count;
}



#endif /* INCLUDE_KITTI_CALIBRATION_COMMON_COMMON_HPP_ */
