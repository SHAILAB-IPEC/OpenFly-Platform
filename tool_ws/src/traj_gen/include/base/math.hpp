#pragma once

#include <Eigen/Dense>
#include <geometry_msgs/Quaternion.h>




double DisFromEigen2Eigen(Eigen::Vector3d first, Eigen::Vector3d second){
    return (second - first).norm();
}


// 将ROS四元数转换为姿态角
Eigen::Vector3d quaternionToEulerAngles(const geometry_msgs::Quaternion& q)
{
    Eigen::Vector3d euler;
    
    // 计算滚转角（绕X轴旋转）
    euler[0] = atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y));

    // 计算俯仰角（绕Y轴旋转）
    euler[1] = asin(2 * (q.w * q.y - q.z * q.x));

    // 计算偏航角（绕Z轴旋转）
    euler[2] = atan2(2 * (q.w* q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));

    return euler;
}

Eigen::Vector3d quaternionToEulerAngles(const Eigen::Quaterniond& q)
{
    Eigen::Vector3d euler;

    euler[0] = atan2(2 * (q.w() * q.x() + q.y() * q.z()), 1 - 2 * (q.x() * q.x() + q.y() * q.y()));

    euler[1] = asin(2 * (q.w() * q.y() - q.z() * q.x()));

    euler[2] = atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));

    return euler;
}
