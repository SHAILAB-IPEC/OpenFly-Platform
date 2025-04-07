#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <future>

#include "traj_gen.cc"
// #include "uavs_net/sendcmd.h"

using namespace std;

bool doFlag = false;
std::string gptcmd_;
Eigen::Vector3d endpoint_ = {0, 0, 0};
bool mission_flag_ = false;
std::vector<Eigen::Vector3d> end_list_;
std::vector<Eigen::Vector3d> mutil_end_list_;
bool mutil_pose_flag_ = false;
bool mutil_flag_ = false;

struct PlannerConfig {
    std::string name;
    int nums;
    double min_dis;
    double max_dis;
    double height_min;
    double height_max;
    int aim_port;
    int listen_port;
    std::string sim_ip;
    int aimlandmark_nums;
    bool add_takeoff_land;
    bool with_turn;

};

std::vector<PlannerConfig> loadthreadconfig(const std::string& config_file){

    YAML::Node config = YAML::LoadFile(config_file);
    std::vector<PlannerConfig> configs;
    if (config["thread_params"]) {
        for (const auto& param : config["thread_params"]) {
            PlannerConfig planner_config;
            planner_config.name = param["name"].as<std::string>();
            planner_config.nums = param["nums"].as<int>();
            planner_config.min_dis = param["min_dis"].as<double>();
            planner_config.max_dis = param["max_dis"].as<double>();
            planner_config.height_min = param["height_min"].as<double>();
            planner_config.height_max = param["height_max"].as<double>();
            planner_config.sim_ip = param["sim_ip"].as<std::string>();
            planner_config.aim_port = param["aim_port"].as<double>();
            planner_config.listen_port = param["listen_port"].as<double>();
            planner_config.aimlandmark_nums = param["aimlandmark_nums"].as<int>();
            planner_config.add_takeoff_land = param["add_takeoff_land"].as<bool>();
            planner_config.with_turn = param["with_turn"].as<bool>();
            configs.push_back(planner_config);
        }
    }
    return configs;
}


void runTrajgenThread(const PlannerConfig& config, std::shared_ptr<rclcpp::Node> node, std::string env_name) {
    TrajGen trajgen(node, config.aim_port, config.listen_port, config.sim_ip, config.name, env_name);
    trajgen.trajDataGen(config.nums, config.min_dis, config.max_dis, config.height_min, config.height_max, config.aimlandmark_nums, config.add_takeoff_land, config.with_turn);
}

std::string getProjectDirectory() {
    std::filesystem::path currentPath(__FILE__);

    std::filesystem::path rosDir = currentPath.parent_path().parent_path().parent_path().parent_path().parent_path();
    
    return rosDir.string(); 
}



int main(int argc, char** argv) {
    rclcpp::init(argc, argv);  // 初始化 ROS 2
    

    std::string config_path_ = getProjectDirectory() + "/configs/";

    std::string env_name = "env_airsim_16";
    if(const char* env = std::getenv("ENV")) {
        env_name = (string)env;
        std::cout << "\033[32m" << "Your ENV is: " << env_name << "\033[0m" << std::endl;;
    }
    else{
        std::cout << "\033[33m" << "Use default environment:" << env_name << "\033[0m" << std::endl;;
    }

    std::string config_path = config_path_  + env_name + ".yaml";

    std::vector<PlannerConfig> configs = loadthreadconfig(config_path);
    std::vector<std::shared_ptr<rclcpp::Node>> nodes;
    for (size_t i = 0; i < configs.size(); ++i) {
        nodes.push_back(std::make_shared<rclcpp::Node>(configs[i].name));
    }
    std::vector<std::thread> threads;
    for (size_t i = 0; i < configs.size(); ++i) {
        threads.emplace_back(runTrajgenThread, configs[i], nodes[i], env_name);
    }
    for (auto& t : threads) {
        t.join();
    }
    rclcpp::shutdown();
    return 0;

}