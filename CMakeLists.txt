cmake_minimum_required(VERSION 2.8.3)
project(fast_livo)

SET(CMAKE_BUILD_TYPE "Debug")

ADD_COMPILE_OPTIONS(-std=c++17 )
set( CMAKE_CXX_FLAGS "-std=c++14 -O3" ) 

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

add_definitions(-w)
# add_definitions(-DUSE_IMG)
# add_definitions(-DUSE_VGICP)
add_definitions(-DMINI)
# add_definitions(-DUSE_ikdtree)
# add_definitions(-DSAVE_PLY)
# add_definitions(-DSAVE_IMG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message("core for MP:  4")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message("core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  livox_ros_driver
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread)
find_package(Ceres REQUIRED)
find_package(GTSAM 4 REQUIRED)
set(Sophus_LIBRARIES libSophus.so)

message(Eigen: ${EIGEN3_INCLUDE_DIR})

include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  include)

add_message_files(
  FILES
  Pose6D.msg
  States.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge image_transport vikit_common vikit_ros
  DEPENDS EIGEN3 PCL OpenCV Sophus
  INCLUDE_DIRS include
)

# set(ONNXRUNTIME_ROOTDIR "/home/zxq/Documents/01ThirdParty/onnxruntime-linux-x64-gpu-1.15.0")
set(ONNXRUNTIME_ROOTDIR "/home/zxq/Documents/09Thirdparty/onnxruntime-linux-x64-1.14.1")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

add_library(std include/STD/STDesc.cpp
                include/LightGlue/LightGlueDecoupleOnnxRunner.cpp
                # include/ikd-Tree/ikd_Tree.cpp
                    )

add_library(vio src/lidar_selection.cpp
                src/frame.cpp
                src/point.cpp
                src/map.cpp
                )
add_executable(fastlivo_mapping src/laserMapping.cpp 
                                src/IMU_Processing.cpp
                                src/preprocess.cpp
                                src/GNSS_Processing.cpp
                                src/laser_mapping.cpp
                                )
target_link_libraries(fastlivo_mapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} ${CERES_LIBRARIES} gtsam onnxruntime vio std)
target_include_directories(fastlivo_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})


