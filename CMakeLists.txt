cmake_minimum_required(VERSION 2.8.3)
project(kitti_calibration)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  image_geometry
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  rostime
  sensor_msgs
  std_msgs
  tf
  kitti
  image_cloud
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES kitti_calibration
  CATKIN_DEPENDS image_geometry pcl_conversions pcl_ros cv_bridge roscpp rospy rostime sensor_msgs std_msgs tf kitti
  DEPENDS system_lib
)

###########
## Build ##
###########

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL 1.7 REQUIRED)

include_directories(BEFORE 	${PCL_INCLUDE_DIRS}  
							${OpenCV_INCLUDE_DIRS}
  							${Boost_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS}
				 ${OpenCV_LIBRARY_DIRS}
)

add_definitions( ${PCL_DEFINITIONS}
				 ${OpenCV_DEFINITIONS}
)
if (NOT PCL_FOUND)
    MESSAGE(FATAL_ERROR "PCL not found.\n")
endif (NOT PCL_FOUND)


## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)

## Declare a cpp library
# add_library(kitti_calibration
#   src/${PROJECT_NAME}/kitti_calibration.cpp
# )

## Declare a cpp executable
# add_executable(kitti_calibration_node src/kitti_calibration_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(kitti_calibration_node kitti_calibration_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(kitti_calibration_node
#   ${catkin_LIBRARIES}
# )

add_executable(gui
  src/${PROJECT_NAME}/gui/node.cpp
  src/${PROJECT_NAME}/gui/gui_opencv.cpp
  src/${PROJECT_NAME}/gui/slider.cpp
)

target_link_libraries(gui
  ${catkin_LIBRARIES}
   ${PCL_LIBRARIES} libvtkCommon.so libvtkFiltering.so
)

add_dependencies(gui
  ${PROJECT_NAME}_gencfg
)


add_executable(benchmark_local
  src/${PROJECT_NAME}/benchmark/benchmark_local.cpp
)

target_link_libraries(benchmark_local
  ${catkin_LIBRARIES}
  kitti
)

add_dependencies(benchmark_local 
  ${PROJECT_NAME}_gencfg
)

add_executable(calibration
  src/${PROJECT_NAME}/calibration/calibration.cpp
)

target_link_libraries(calibration
  ${catkin_LIBRARIES}
  kitti
)

add_dependencies(calibration 
  ${PROJECT_NAME}_gencfg
)

## Create image samples, benchmark times
add_executable(create_samples
  src/${PROJECT_NAME}/benchmark/create_samples.cpp
)

target_link_libraries(create_samples
  ${catkin_LIBRARIES}
  kitti
)

add_dependencies(create_samples
  ${PROJECT_NAME}_gencfg
)

## Create image samples, benchmark times
add_executable(filter
  src/${PROJECT_NAME}/filter/filter.cpp
)

target_link_libraries(filter
  ${catkin_LIBRARIES}
  kitti
)

add_dependencies(filter
  ${PROJECT_NAME}_gencfg
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS kitti_calibration kitti_calibration_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_kitti_calibration.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
