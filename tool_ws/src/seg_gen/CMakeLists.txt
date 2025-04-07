cmake_minimum_required(VERSION 3.5)
project(seg_gen)

add_definitions(-DROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(yaml-cpp REQUIRED) 
find_package(nlohmann_json REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

include_directories(
  include
  ${Eigen_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})

add_executable(bev_seg_node src/bev_seg_node.cc)
ament_target_dependencies(bev_seg_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs 
  cv_bridge
  Eigen3
  yaml-cpp
  pcl_ros 
  nlohmann_json 
)
target_link_libraries(bev_seg_node jsoncpp glog nlohmann_json::nlohmann_json yaml-cpp)
install(TARGETS bev_seg_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


add_executable(manual_seg_node src/manual_seg_node.cc)
ament_target_dependencies(manual_seg_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs 
  cv_bridge
  Eigen3
  yaml-cpp
  pcl_ros 
  nlohmann_json 
)
target_link_libraries(manual_seg_node jsoncpp glog nlohmann_json::nlohmann_json yaml-cpp)
install(TARGETS manual_seg_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
