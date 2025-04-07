#pragma once

#include <Eigen/Dense>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Path.h>



Eigen::Quaterniond geometryToEigen(const geometry_msgs::Quaternion& vec_ros)
{
  Eigen::Quaterniond vec_eigen;
  vec_eigen.x() = vec_ros.x;
  vec_eigen.y() = vec_ros.y;
  vec_eigen.z() = vec_ros.z;
  vec_eigen.w() = vec_ros.w;
  return vec_eigen;
}

geometry_msgs::Quaternion eigenToGeometry(const Eigen::Quaterniond& vec_eigen)
{
  geometry_msgs::Quaternion vec_ros;
  vec_ros.x = vec_eigen.x();
  vec_ros.y = vec_eigen.y();
  vec_ros.z = vec_eigen.z();
  vec_ros.w = vec_eigen.w();
  return vec_ros;
}

//Vectors and Points
Eigen::Vector3d geometryToEigen(const geometry_msgs::Vector3& vec_ros)
{
  Eigen::Vector3d vec_eigen;
  vec_eigen.x() = vec_ros.x;
  vec_eigen.y() = vec_ros.y;
  vec_eigen.z() = vec_ros.z;
  return vec_eigen;
}

Eigen::Vector3d geometryToEigen(const geometry_msgs::Point& vec_ros)
{
  Eigen::Vector3d vec_eigen;
  vec_eigen.x() = vec_ros.x;
  vec_eigen.y() = vec_ros.y;
  vec_eigen.z() = vec_ros.z;
  return vec_eigen;
}

geometry_msgs::Vector3 eigenToGeometry(const Eigen::Vector3d& vec_eigen)
{
  geometry_msgs::Vector3 vec_ros;
  vec_ros.x = vec_eigen.x();
  vec_ros.y = vec_eigen.y();
  vec_ros.z = vec_eigen.z();
  return vec_ros;
}

geometry_msgs::Point vectorToPoint(const geometry_msgs::Vector3& vector)
{
  geometry_msgs::Point point;
  point.x = vector.x;
  point.y = vector.y;
  point.z = vector.z;
  return point;
}

Eigen::Affine3d geometryToEigen(const geometry_msgs::Pose& pose_ros)
{
  Eigen::Affine3d pose;
  Eigen::Vector3d translation(pose_ros.position.x, pose_ros.position.y,
                              pose_ros.position.z);
  Eigen::Quaterniond rotation(pose_ros.orientation.w, pose_ros.orientation.x,
                              pose_ros.orientation.y, pose_ros.orientation.z);

  pose = Eigen::Translation3d(translation) * rotation;
  return pose;
}

std::vector<Eigen::Vector3d> PathToEigen(const nav_msgs::Path& path)
{
  std::vector<Eigen::Vector3d> path_eigen;
  for (const auto& pose : path.poses)
  {
    path_eigen.push_back(geometryToEigen(pose.pose.position));
  }
  return path_eigen;
}


