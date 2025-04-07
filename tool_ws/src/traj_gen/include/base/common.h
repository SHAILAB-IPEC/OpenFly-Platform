/**
 * @file base.h
 * @author WangLiansheng (wangliansheng@pjlab.org.cn)
 * @brief 
 * @version 0.1
 * @date 2023-06-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef BASE_H_H_H_
#define BASE_H_H_H_


#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include "base/eigen_types.h"


enum UavState {
  DISARM = 1,
  ARM,
  LAND,       // 3 
  TAKEOFF,
  HOVER,
  FLY
};

/* 无人的飞行状态 */
struct State
{
  Eigen::Vector3d pt;      // 全局的位置
  Eigen::Vector3d vel;     // 全局的速度  
  Eigen::Vector3d acc;     // body
  Eigen::Vector3d rpy;     // Yaw是全局的，r、p body

  State(){
    pt  = Eigen::Vector3d::Zero();
    vel = Eigen::Vector3d::Zero();
    acc = Eigen::Vector3d::Zero();
    rpy = Eigen::Vector3d::Zero();
  }
};


#endif