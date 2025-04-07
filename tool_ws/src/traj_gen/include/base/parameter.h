/**
 * @file parameter.h
 * @author WangLiansheng (wangliansheng@pjlab.org.cn)
 * @brief 
 * @version 0.1
 * @date 2023-06-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef PARAMETER_H_WLS_
#define PARAMETER_H_WLS_
#include <atomic>
#include <string>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>
using namespace std;

#define MAX_VEL 2.0
#define MAX_W   0.5
#define DEFAULT_HEIGHT 1.0

std::string g_root_dir = std::string(ROOT);

namespace Para {
  /* logs */
  inline bool g_save_logs = false;

  /* ROS topic */
  inline std::string g_gpt_srv_topic = "/uav_gpt_srv";
  inline std::string g_uav_odom_topic = "/mavros/local_position/pose";
  inline std::string g_global_map_topic = "/visual/global_map";

  /* plan env */
  inline std::string g_map_name = "pjlab.pcd";

  inline double g_max_vel = 1.5;
  inline double g_max_w = 0.34;        
  inline double g_max_height = 1.8;      

  /* 无人机固有属性 */
  inline double g_MaxVel = 1.5;
  inline double g_MaxAcc = 1.0;
  inline double g_MaxYawRate = M_PI;

  void loadParams(ros::NodeHandle& nh) {
    nh.param<bool>("/logs/save_logs", g_save_logs, false);

    nh.param<std::string>("/ros/gpt_srv_topic", g_gpt_srv_topic, "/gpt_srv");
    nh.param<std::string>("/ros/uav_odom_topic", g_uav_odom_topic, "/mavros/local_position/pose");
    nh.param<std::string>("/ros/global_map_topic", g_global_map_topic, "/visual/global_map");

    nh.param<std::string>("/plan_env/map_name", g_map_name, "pjlab.pcd");

    nh.param<double>("/motion_constraints/max_vel", g_max_vel, 1.5);
    nh.param<double>("/motion_constraints/max_w", g_max_w, 0.34);
    nh.param<double>("/motion_constraints/max_height", g_max_height, 1.8);

    nh.param<double>("/uav_para/MaxVel", g_MaxVel, 1.5);
    nh.param<double>("/uav_para/MaxAcc", g_MaxAcc, 1.0);
    nh.param<double>("/uav_para/MaxYawRate", g_MaxYawRate, M_PI);
  }

  void loadParams(const std::string& file_path) {
    YAML::Node config = YAML::LoadFile(file_path);
    // logs
    g_save_logs = config["logs"]["save_logs"].as<bool>(false);

    // ROS topic
    g_gpt_srv_topic = config["ros"]["gpt_srv_topic"].as<std::string>("/gpt_srv");
    g_uav_odom_topic = config["ros"]["uav_odom_topic"].as<std::string>("/mavros/local_position/pose");
    g_global_map_topic = config["ros"]["global_map_topic"].as<std::string>("/visual/global_map");

    // Plan env
    g_map_name = config["plan_env"]["map_name"].as<std::string>("pjlab.pcd");

    // motion constraints
    g_max_vel = config["motion_constraints"]["max_vel"].as<double>(1.5);
    g_max_w = config["motion_constraints"]["max_w"].as<double>(0.34);
    g_max_height = config["motion_constraints"]["max_height"].as<double>(1.8);

    // UAV inherent properties
    g_MaxVel = config["uav_para"]["MaxVel"].as<double>(1.5);
    g_MaxAcc = config["uav_para"]["MaxAcc"].as<double>(1.0);
    g_MaxYawRate = config["uav_para"]["MaxYawRate"].as<double>(M_PI);
  }
}


// run time --> RT
namespace RT{
  inline bool FLAG_EXIT = false;
};

#endif