#include <iostream>
#include <thread>
#include <chrono>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <random>
#include <cstdlib>

// ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

// ROS2 msgs
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// ROS2 package
#include <ament_index_cpp/get_package_share_directory.hpp>

// Eigen
#include <Eigen/Eigen>

// sys
#include <sys/socket.h>
#include <arpa/inet.h>
#include <yaml-cpp/yaml.h>
#include <json/json.h>
#include <nlohmann/json.hpp>

// PCL
#include <pcl/common/transforms.h>

//lib
#include "map/voxel_map.hpp"
#include "map/loadpcdmap.hpp"
#include "navi/pathsearch.hpp"
#include "base/tcpserver.hpp"
#include "map/seggridmap.hpp"
#include "common.h"


namespace fs = std::filesystem;

class TrajGen{
private:
    //ros
    std::shared_ptr<rclcpp::Node> node_; 
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_Pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr search_path_Pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr show_point_Pub_;

    //navi
    std::vector<Eigen::Vector3d> global_route_;
    nav_msgs::msg::Path true_path_;
    Eigen::Vector3d target_;
    PathSearch* pathsearch_;
    double flyheight_;

    //map
    voxel_map::VoxelMap global_voxelmap_;
    voxel_map::VoxelMap bev_voxelmap_;
    double dilateRadius_;
    double voxelWidth_;
    std::vector<double> mapBound_;
    sensor_msgs::msg::PointCloud2 cloud_msg_;
    bool mapInitialized_;
    std::vector<Eigen::Vector3d> eigen_pc_;


    //tcp
    TCPServer* tcpserver_;
    int listen_port_;
    int send_port_; 
    std::string sim_ip_;


    // Visualizer visualizer;
    std::string data_record_path_ = "";
    std::string pose_record_file_name_;
    int image_frame_ = 0;

    //scene data
    std::string env_;
    std::string data_type_;
    std::string pr_dir_;
    double send_freq_;
    double env_scale_ratio_;    
    double traj_scale_ratio_;
    double map_elevation_;
    double min_height_thresh_;
    double objinfov_height_thr_ = 15;
    double map_height_thr_ = 0;

    std::vector<Object> obj_list_;
    Object cur_obj_;

public:
    TrajGen(std::shared_ptr<rclcpp::Node> node, int send_p, int listen_p, std::string sim_ip,
                    const std::string& node_name, const std::string& env_name)
        : node_(node), 
        mapInitialized_(false), 
        send_port_(send_p), 
        listen_port_(listen_p),
        sim_ip_(sim_ip), 
        env_(env_name)
    {
        global_map_Pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(node_name + "/trajgen/show/global_map", 10);
        search_path_Pub_ = node_->create_publisher<nav_msgs::msg::Path>(node_name + "/trajgen/show/search_path", 10);
        show_point_Pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(node_name + "/trajgen/show/points", 10);
        pr_dir_ = getProjectDir();
        std::string yaml_path = pr_dir_ +  "/configs/" + env_ + ".yaml";
        std::cout << "\033[33m"  << "The Root Directory of OpenFly Project: " << pr_dir_  << "\033[0m" << std::endl;
        std::cout << "\033[33m" << "The YAML file path: " << yaml_path << "\033[0m" << std::endl;
        // load YAML configuration
        loadYaml(yaml_path);
        data_record_path_ = pr_dir_ + "/uav_vln_data/" + env_ + "/" + data_type_ + "/";
        mapBound_[4] = map_elevation_;
        const Eigen::Vector3i xyz(
            (mapBound_[1] - mapBound_[0]) / voxelWidth_,
            (mapBound_[3] - mapBound_[2]) / voxelWidth_,
            (mapBound_[5] - mapBound_[4]) / voxelWidth_
        );
        const Eigen::Vector3d offset(mapBound_[0], mapBound_[2], mapBound_[4]);
        const Eigen::Vector3i bev_xyz(
            (mapBound_[1] - mapBound_[0]) / voxelWidth_,
            (mapBound_[3] - mapBound_[2]) / voxelWidth_,
            voxelWidth_ / voxelWidth_
        );
        const Eigen::Vector3d bev_offset(mapBound_[0], mapBound_[2], -0.5 * voxelWidth_);
        global_voxelmap_ = voxel_map::VoxelMap(xyz, offset, voxelWidth_);
        bev_voxelmap_ = voxel_map::VoxelMap(bev_xyz, bev_offset, voxelWidth_);
        // Build the global and BEV maps, and initialize the graph
        globalMapBulid();
        bevMapBulid();
        initGraph();

        pathsearch_ = new PathSearch(global_voxelmap_, node_);
        tcpserver_ = new TCPServer(sim_ip_, send_port_, listen_port_);

        std::cout << "Initialization complete." << std::endl;
    }

    std::string getProjectDir() {
        fs::path currentPath(__FILE__);
        fs::path rosDir = currentPath.parent_path().parent_path().parent_path().parent_path().parent_path();
        return rosDir.string(); 
    }

    void trajDataGen(int k, int r, int R, int h, int H, int landmark_nums = 1, bool add_takeoff_land = false, bool with_turn = false) {
        if (env_.find("gs") != std::string::npos) {
            add_takeoff_land = false;
            with_turn = false;
            std::cout << "\033[33m" << "GS env cannot add takeoff and landing" << "\033[0m"<< std::endl;
            std::cout << "\033[33m" << "GS env cannot forcefully navigate obstacles on a single trajectory" << "\033[0m"<< std::endl;
        }
        std::cout << "\033[32m" << "***************************************"                                                                            << "\033[0m" << std::endl;
        std::cout << "\033[32m" << "Plan to collect " << k << " trajectories"                                                                           << "\033[0m" << std::endl;
        std::cout << "\033[32m" << "Each trajectory has " << landmark_nums << " landmarks"                                                              << "\033[0m" << std::endl;
        std::cout << "\033[32m" << (add_takeoff_land ? "with takeoff and landing" : " without takeoff and landing")                                     << "\033[0m" << std::endl;
        std::cout << "\033[32m" << (with_turn ? "There must be a turn in the trajectory" : "There may not necessarily be a turn in the trajectory")     << "\033[0m" << std::endl;
        std::cout << "\033[32m" << "***************************************"                                                                            << "\033[0m" << std::endl;
        if(landmark_nums == 1){
            genTrajbyOneLandmark(k, r, R, h, H, add_takeoff_land, with_turn);
        }
        else if (landmark_nums >= 1)
        {
            genTrajbyMultiLandmark(k, r, R, h, H, landmark_nums , with_turn, add_takeoff_land);
        }
    }

    void globalMapBulid() {
        cloud_msg_ = loadpcdfile(pr_dir_ + "/scene_data/pcd_map/" + env_ + ".pcd");
        cloud_msg_ = scalePointCloud(cloud_msg_, env_scale_ratio_);
        cloud_msg_ = filterPointCloud(cloud_msg_, voxelWidth_);
        cloud_msg_ = removeOutliersWithStatisticalFilter(cloud_msg_, 60, 2.0);  

        cloud_msg_.header.frame_id = "map";
        cloud_msg_.header.stamp = node_->get_clock()->now();
        rclcpp::sleep_for(std::chrono::seconds(2));
        global_map_Pub_->publish(cloud_msg_);
        std::cout << "\033[32m" <<"Global map loaded successfully."  << "\033[0m" << std::endl;

        if (!mapInitialized_) {
            size_t cur = 0;
            const size_t total = cloud_msg_.data.size() / cloud_msg_.point_step;
            float *fdata = reinterpret_cast<float*>(&cloud_msg_.data[0]);
            for (size_t i = 0; i < total; i++) {
                cur = cloud_msg_.point_step / sizeof(float) * i;
                if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                    std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                    std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2])) {
                    continue;
                }
                eigen_pc_.push_back(Eigen::Vector3d(fdata[cur + 0], fdata[cur + 1], fdata[cur + 2]));
                global_voxelmap_.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                            fdata[cur + 1],
                                                            fdata[cur + 2]));
            }
            global_voxelmap_.dilate(std::ceil(dilateRadius_ / global_voxelmap_.getScale()));
            mapInitialized_ = true;
        }
    }

    void bevMapBulid() {
        size_t cur = 0;
        const size_t total = cloud_msg_.data.size() / cloud_msg_.point_step;
        float *fdata = reinterpret_cast<float*>(&cloud_msg_.data[0]);
        for (size_t i = 0; i < total; i++) {
            cur = cloud_msg_.point_step / sizeof(float) * i;
            if (std::isnan(fdata[cur + 0]) || std::isinf(fdata[cur + 0]) ||
                std::isnan(fdata[cur + 1]) || std::isinf(fdata[cur + 1]) ||
                std::isnan(fdata[cur + 2]) || std::isinf(fdata[cur + 2])) {
                continue;
            }
            if(fdata[cur + 2]  < map_elevation_ + min_height_thresh_) continue;
            bev_voxelmap_.setOccupied(Eigen::Vector3d(fdata[cur + 0],
                                                            fdata[cur + 1],
                                                            0));
        }
        bev_voxelmap_.dilate(std::ceil(dilateRadius_ / bev_voxelmap_.getScale()));
        mapInitialized_ = true;   
    }

    bool bevQuery(Eigen::Vector3d pos){
        pos.z() = 0;
        if(bev_voxelmap_.query(pos)){
            return true;
        }
        else return false;
    }

    void initGraph(){
        std::string graph_json = pr_dir_ + "/scene_data/seg_map/" + env_ +  ".jsonl";
        obj_list_ = loadObjectsFromFile(graph_json);
    }

    Object parseObject(const nlohmann::json& jsonObject) {

        std::string type = jsonObject.at("type").get<std::string>();
        std::string color = jsonObject.at("color").get<std::string>();
        std::string size = jsonObject.at("size").get<std::string>();
        std::string shape = jsonObject.at("shape").get<std::string>();
        std::string feature = jsonObject.at("feature").get<std::string>();
        std::string filename = jsonObject.at("filename").get<std::string>();

        double x, y, z;
        sscanf(filename.c_str(), "X=%lfY=%lfZ=%lf", &x, &y, &z);
        Eigen::Vector3d pos(x, y, z);
        if(env_ == "env_ue_bigcity" || env_ =="env_ue_smallcity"){
            pos = pos/100; 
            pos[1] = -pos[1];
        }
        else{
            pos = pos * env_scale_ratio_;
        }
        return Object(type, color, size, shape, feature, pos);
    }


    std::vector<Object> loadObjectsFromFile(const std::string& filePath) {
        std::vector<Object> objects;
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "Failed to open JSON file: " << filePath << std::endl;
            return objects;
        }
        std::string line;
        while (std::getline(file, line)) {
            try {
                nlohmann::json jsonObject = nlohmann::json::parse(line);
                objects.push_back(parseObject(jsonObject));
            } catch (const nlohmann::json::exception& e) {
                std::cerr << "JSON parsing error: " << e.what() << std::endl;
            }
        }
        file.close();
        return objects;
    }


    void loadYaml(std::string yaml_file){
        std::ifstream ifile(yaml_file);
        if (!ifile) {
            std::cerr << "YAML file not found: " << yaml_file << std::endl;
            return;
        }
            auto yaml = YAML::LoadFile(yaml_file);
            env_ = yaml["datagen"]["env"].as<std::string>();
            data_type_ = yaml["datagen"]["data_type"].as<std::string>();
            send_freq_ = yaml["datagen"]["freq"].as<double>();
            dilateRadius_ = yaml["traj_map"]["DilateRadius"].as<double>();
            voxelWidth_ = yaml["traj_map"]["VoxelWidth"].as<double>();
            mapBound_ = {
                yaml["traj_map"]["MapBound"][0].as<double>(),
                yaml["traj_map"]["MapBound"][1].as<double>(),
                yaml["traj_map"]["MapBound"][2].as<double>(),
                yaml["traj_map"]["MapBound"][3].as<double>(),
                yaml["traj_map"]["MapBound"][4].as<double>(),
                yaml["traj_map"]["MapBound"][5].as<double>()
            };
            env_scale_ratio_ = yaml["traj_map"]["pcd_scale_ratio"].as<double>();
            traj_scale_ratio_ = yaml["traj_map"]["traj_scale_ratio"].as<double>();
            map_elevation_ = yaml["traj_map"]["map_elevation"].as<double>();
            min_height_thresh_ = yaml["traj_map"]["min_height_thresh"].as<double>();
            map_height_thr_ = map_elevation_ + min_height_thresh_;

    }
 
    std::string getCurrentTimeString() {
        std::time_t now = std::time(nullptr);
        std::tm *ltm = std::localtime(&now);
        std::ostringstream time_stream;
        time_stream << 1900 + ltm->tm_year << "-"
                    << 1 + ltm->tm_mon << "-"
                    << ltm->tm_mday << "_"
                    << ltm->tm_hour << "-"
                    << ltm->tm_min << "-"
                    << ltm->tm_sec;
        int random_number = std::rand();
        time_stream << "_" << random_number; 
        return time_stream.str();
    }

    void initRecord(const std::string& filename) {
        std::string time_path = getCurrentTimeString();
        pose_record_file_name_ = filename + time_path + "/";
        try {
            fs::create_directories(pose_record_file_name_);
            std::cout << "Directory created for record pose : " << pose_record_file_name_ << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error creating directories: " << e.what() << std::endl;
        }
    }


    void astartSearchPath(Eigen::Vector3d start, Eigen::Vector3d end, bool with_stop = false){ 
        true_path_.poses.clear();
        std::vector<Eigen::Vector3d> v3d_path;
        sleep(2);
        v3d_path = pathsearch_->hybridAStar(start, end);
        nav_msgs::msg::Path showpath;
        showpath.header.frame_id = "map";
        showpath.header.stamp = node_->now();

        for (const auto& point : v3d_path) {
            geometry_msgs::msg::PoseStamped tmp_pose;
            tmp_pose.pose.position.x = point[0];
            tmp_pose.pose.position.y = point[1];
            tmp_pose.pose.position.z = point[2];
            showpath.poses.push_back(tmp_pose);
        }
        rclcpp::Rate looprate(10);
        for (int i = 0; i < 10; i++) {
            search_path_Pub_->publish(showpath);
            looprate.sleep();
        }
        pathsearch_->backtrackpath(v3d_path, with_stop);
    }


    void saveTrajListToJson(std::vector<TrajRecodData> traj_list, const std::string& filename){
        std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> action_list;
        std::vector<Eigen::Vector4d> record_list;
        using json = nlohmann::json;
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            std::cerr << "\033[31m" << "Failed to open file for writing : " << filename  << "\033[0m"<< std::endl;
            return;
        }
        int image_id = 0;

        for(int i = 0 ; i < traj_list.size(); i ++){
            action_list = traj_list[i].action_list;
            record_list = traj_list[i].record_list;

            for(size_t i = 0 ; i < action_list.size(); i++){
                json j;

                json actionj;
                actionj["imageid"] = image_id;
                actionj["type"] = action_list[i].first.first;
                actionj["value"] = action_list[i].first.second;
                actionj["pos"] = action_list[i].second;
                actionj["yaw"] = record_list[i][3];
                j["action"] = actionj;
                outputFile << j.dump(4) << std::endl; 
                image_id ++;
            }
            Object obj = traj_list[i].aim_obj;
            json jendobj;
            jendobj["type"] = obj.type;
            jendobj["color"] = obj.color;
            jendobj["size"] = obj.size;
            jendobj["shape"] = obj.shape;
            jendobj["feature"] = obj.feature;
            jendobj["position"] = {obj.pos.x(), obj.pos.y(), obj.pos.z()};  // Convert Eigen::Vector3d to array
            json aim_landmark_j;
            aim_landmark_j["aim_landmark"] = jendobj;
            outputFile << aim_landmark_j.dump(4) << std::endl;  // Pretty print with indent of 4
        }
        outputFile.close();
        
    }

    void saveActionsToJson(const std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>>& action_list, 
                            std::vector<Eigen::Vector4d> record_list,
                            const std::string& filename) {
        using json = nlohmann::json;
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            std::cerr << "\033[31m" << "Failed to open file for writing : " << filename  << "\033[0m"<< std::endl;
            return;
        }
        for(size_t i = 0 ; i < action_list.size(); i++){
            json j;

            json actionj;
            actionj["imageid"] = i;
            actionj["type"] = action_list[i].first.first;
            actionj["value"] = action_list[i].first.second;
            actionj["pos"] = action_list[i].second;
            actionj["yaw"] = record_list[i][3];
            j["action"] = actionj;
            outputFile << j.dump(4) << std::endl; // 每个 JSON 对象占一行
        }

        Object obj = cur_obj_;
        json jendobj;
        jendobj["type"] = obj.type;
        jendobj["color"] = obj.color;
        jendobj["size"] = obj.size;
        jendobj["shape"] = obj.shape;
        jendobj["feature"] = obj.feature;
        jendobj["position"] = {obj.pos.x(), obj.pos.y(), obj.pos.z()};  // Convert Eigen::Vector3d to array
        
        json aim_landmark_j;
        aim_landmark_j["aim_landmark"] = jendobj;
        outputFile << aim_landmark_j.dump(4) << std::endl;  // Pretty print with indent of 4
        outputFile.close();
    }


    void recordPathbyTCP(std::vector<TrajRecodData> traj_list, bool add_takeoff_land){    
        image_frame_ = 1;
        initRecord(data_record_path_);
        tcpserver_->sendString("path:"+ pose_record_file_name_);
        sleep(1.0);
        std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> action_list;
        std::vector<Eigen::Vector4d> record_list;
        for(int i = 0 ; i < traj_list.size() - 1; i++){
            double yaw_error = traj_list[i + 1].record_list[0][3] - traj_list[i].record_list.back()[3];
            if(abs(yaw_error) < 0.01){continue;}
            if(yaw_error > M_PI) yaw_error -=  2 * M_PI;
            else if(yaw_error < -M_PI) yaw_error += 2 * M_PI;
            int turn_nums = abs(yaw_error) / (M_PI/6);
            double step_yaw = (M_PI) / 6;
            double in_yaw = traj_list[i].record_list.back()[3];
            Eigen::Vector3d cur_p(traj_list[i].record_list.back().head<3>());
            if (yaw_error != 0) {
                if (yaw_error > 0) {
                    traj_list[i].action_list.back().first.first = "turn left";
                    for(int j = 1 ; j < turn_nums; j ++){
                        action_list.emplace_back(std::make_pair(std::make_pair("turn left", 30), cur_p));
                        record_list.emplace_back(Eigen::Vector4d(cur_p[0], cur_p[1], cur_p[2], in_yaw + j * step_yaw));
                    }
                } else {
                    traj_list[i].action_list.back().first.first = "turn right";
                    for(int j = 1 ; j < turn_nums; j ++){
                        action_list.emplace_back(std::make_pair(std::make_pair("turn right", 30), cur_p));
                        record_list.emplace_back(Eigen::Vector4d(cur_p[0], cur_p[1], cur_p[2], in_yaw - j * step_yaw));
                    }
                }
            }
            traj_list[i + 1].action_list.insert(traj_list[i + 1].action_list.begin() , action_list.begin(), action_list.end());
            traj_list[i + 1].record_list.insert(traj_list[i + 1].record_list.begin(), record_list.begin(), record_list.end());
        }
        if(add_takeoff_land){
            addTakeoffLanding(traj_list);
        }
        for(int i = 0 ; i < traj_list.size(); i++){
            std::vector<Eigen::Vector4d> record_list = traj_list[i].record_list;
            if(record_list.size() < 1) continue;    
            for(int j = 0 ; j < record_list.size(); j ++){
                double yaw = record_list[j][3];
                Eigen::Vector3d pos = record_list[j].head<3>();
                pos = pos * (traj_scale_ratio_ /env_scale_ratio_);
                yaw = yaw  / M_PI * 180;
                if(abs(yaw) < 0.1) yaw = 0;
                tcpserver_->sendCameraPose(pos[0], pos[1], pos[2], 0 ,yaw ,0);
                sleep(1.0/send_freq_);
                if(j == 0 && i == 0){
                    if(env_.find("gs") != std::string::npos){
                        tcpserver_->sendCameraPose(pos[0], pos[1], pos[2], 0 , yaw ,0);
                    }
                    sleep(1);
                }
            }
        }
        saveTrajListToJson(traj_list, pose_record_file_name_ + "pose.jsonl");
    }

    double getMaxZinRange(const std::vector<Eigen::Vector3d>& filtered_points) {
        double max_z = std::numeric_limits<double>::lowest();  
        for (const auto& point : filtered_points) {
            double z = point[2];
            if (z > max_z) {
                max_z = z;
            }
        }
        return max_z;
    }

    double getMaxZinP(double input_x , double input_y, double range){
        double min_x = input_x - range;
        double max_x = input_x + range;
        double min_y = input_y - range;
        double max_y = input_y + range;
        const std::vector<Eigen::Vector3d> point_cloud = eigen_pc_;
        std::vector<Eigen::Vector3d> filtered_points = filterPointsinRange(point_cloud, min_x, max_x, min_y, max_y);
        double max_z = std::numeric_limits<double>::lowest();  
        for (const auto& point : filtered_points) {
            double z = point[2];
            if (z > max_z) {
                max_z = z;
            }
        }
        return max_z;
    }

    double calDis(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
        return std::sqrt(std::pow(p2[0] - p1[0], 2) + std::pow(p2[1] - p1[1], 2) + std::pow(p2[2] - p1[2], 2));
    }

    void addTakeoffLanding(
        std::vector<TrajRecodData>& traj_list,
        double range_radius = 2.0, 
        double distance_interval = 3.0) 
    {
        std::vector<Eigen::Vector3d> point_cloud = eigen_pc_;
        Eigen::Vector4d start_point = traj_list.front().record_list.front();  
        Eigen::Vector4d end_point = traj_list.back().record_list.back();   
        
        double min_x_takeoff = start_point[0] - range_radius;
        double max_x_takeoff = start_point[0] + range_radius;
        double min_y_takeoff = start_point[1] - range_radius;
        double max_y_takeoff = start_point[1] + range_radius;

        double min_x_landing = end_point[0] - range_radius;
        double max_x_landing = end_point[0] + range_radius;
        double min_y_landing = end_point[1] - range_radius;
        double max_y_landing = end_point[1] + range_radius;

        std::vector<Eigen::Vector3d> filtered_takeoff_points = filterPointsinRange(point_cloud, min_x_takeoff, max_x_takeoff, min_y_takeoff, max_y_takeoff);
        double takeoff_height = getMaxZinRange(filtered_takeoff_points);  
        std::vector<Eigen::Vector3d> filtered_landing_points = filterPointsinRange(point_cloud, min_x_landing, max_x_landing, min_y_landing, max_y_landing);
        double landing_height = getMaxZinRange(filtered_landing_points);  

        if(landing_height < map_elevation_){
            landing_height = map_elevation_;
        }
        if(takeoff_height < map_elevation_){
            takeoff_height = map_elevation_;
        }
        if(landing_height > end_point[2]){
            landing_height = end_point[2];
        }
        if(takeoff_height > start_point[2]){
            takeoff_height = start_point[2];
        }

        Eigen::Vector3d takeoff_start(start_point[0], start_point[1], takeoff_height);
        Eigen::Vector3d takeoff_end(start_point[0], start_point[1], start_point[2]);
        double distance_to_travel = calDis(takeoff_start, takeoff_end);
        int num_points_to_insert_takeoff = static_cast<int>(std::floor(distance_to_travel / distance_interval));
        
        for (int i = 1; i <= num_points_to_insert_takeoff; ++i) {
            double new_height = start_point[2] - i * distance_interval;
            Eigen::Vector4d takeoff_point(start_point[0], start_point[1], new_height, start_point[3]);
            traj_list.front().record_list.insert(traj_list.front().record_list.begin(), takeoff_point);  // 插入到起始位置
            traj_list.front().action_list.insert(traj_list.front().action_list.begin(), {{"up", 3}, Eigen::Vector3d(takeoff_start[0], takeoff_start[1], new_height)});
        }

        Eigen::Vector3d landing_start(end_point[0], end_point[1], landing_height);
        Eigen::Vector3d landing_end(end_point[0], end_point[1], end_point[2]);
        distance_to_travel = calDis(landing_start, landing_end);
        int num_points_to_insert_landing = static_cast<int>(std::floor(distance_to_travel / distance_interval));

        traj_list.back().action_list.back().first.first = "down";
        for (int i = 1; i <= num_points_to_insert_landing; ++i) {
            double new_height = end_point[2] - i * distance_interval;
            Eigen::Vector4d landing_point(end_point[0], end_point[1], new_height, end_point[3]);
            traj_list.back().record_list.push_back(landing_point);  
            traj_list.back().action_list.push_back({{"down", 3}, Eigen::Vector3d(landing_start[0], landing_start[1], new_height)});
        }
        traj_list.back().action_list.back().first.first = "stop";
    }

    void addTakeoffLanding(
        std::vector<Eigen::Vector4d>& record_list,
        std::vector<std::pair<std::pair<std::string, int>, Eigen::Vector3d>>& action_list,
        double range_radius = 2.0, 
        double distance_interval = 3.0) {

        std::vector<Eigen::Vector3d> point_cloud = eigen_pc_;
        Eigen::Vector4d start_point = record_list.front();  
        Eigen::Vector4d end_point = record_list.back();   
        
        double min_x_takeoff = start_point[0] - range_radius;
        double max_x_takeoff = start_point[0] + range_radius;
        double min_y_takeoff = start_point[1] - range_radius;
        double max_y_takeoff = start_point[1] + range_radius;

        double min_x_landing = end_point[0] - range_radius;
        double max_x_landing = end_point[0] + range_radius;
        double min_y_landing = end_point[1] - range_radius;
        double max_y_landing = end_point[1] + range_radius;

        std::vector<Eigen::Vector3d> filtered_takeoff_points = filterPointsinRange(point_cloud, min_x_takeoff, max_x_takeoff, min_y_takeoff, max_y_takeoff);
        double takeoff_height = getMaxZinRange(filtered_takeoff_points);  
        std::vector<Eigen::Vector3d> filtered_landing_points = filterPointsinRange(point_cloud, min_x_landing, max_x_landing, min_y_landing, max_y_landing);
        double landing_height = getMaxZinRange(filtered_landing_points); 

        if(landing_height < map_elevation_){
            landing_height = map_elevation_;
        }
        if(takeoff_height < map_elevation_){
            takeoff_height = map_elevation_;
        }
        if(landing_height > end_point[2]){
            landing_height = end_point[2];
        }
        if(takeoff_height > start_point[2]){
            takeoff_height = start_point[2];
        }

        Eigen::Vector3d takeoff_start(start_point[0], start_point[1], takeoff_height);
        Eigen::Vector3d takeoff_end(start_point[0], start_point[1], start_point[2]);

        double distance_to_travel = calDis(takeoff_start, takeoff_end);
        int num_points_to_insert_takeoff = static_cast<int>(std::floor(distance_to_travel / distance_interval));
        
        for (int i = 1; i <= num_points_to_insert_takeoff; ++i) {
            double new_height = start_point[2] - i * distance_interval;
            Eigen::Vector4d takeoff_point(start_point[0], start_point[1], new_height, start_point[3]);
            record_list.insert(record_list.begin(), takeoff_point); 
            action_list.insert(action_list.begin(), {{"up", 3}, Eigen::Vector3d(takeoff_start[0], takeoff_start[1], new_height)});
        }
        Eigen::Vector3d landing_start(end_point[0], end_point[1], landing_height);
        Eigen::Vector3d landing_end(end_point[0], end_point[1], end_point[2]);
        distance_to_travel = calDis(landing_start, landing_end);
        int num_points_to_insert_landing = static_cast<int>(std::floor(distance_to_travel / distance_interval));

        action_list.back().first.first = "down";
        for (int i = 1; i <= num_points_to_insert_landing; ++i) {
            double new_height = end_point[2] - i * distance_interval;
            Eigen::Vector4d takeoff_point(end_point[0], end_point[1], new_height, end_point[3]);
            record_list.push_back(takeoff_point); 
            action_list.push_back({{"down", 3}, Eigen::Vector3d(landing_start[0], landing_start[1], new_height)});
        }
        action_list.back().first.first = "stop";
    }


    void recordPathbyTCP(bool is_takeoff_landing){
        std::vector<Eigen::Vector4d> record_list;
        std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> action_list;
        
        record_list = pathsearch_->getrecordlist();
        action_list = pathsearch_->getactionlist();

        image_frame_ = 1;
        if(record_list.size() < 1) return;
        initRecord(data_record_path_);
        tcpserver_->sendString("path:"+ pose_record_file_name_);
        sleep(0.2);
        if(is_takeoff_landing){
            addTakeoffLanding(record_list, action_list);
        }
        for(int i = 0 ; i < record_list.size(); i ++){
            double yaw = record_list[i][3];
            Eigen::Vector3d pos = record_list[i].head<3>();
            pos = pos * (traj_scale_ratio_ /env_scale_ratio_);
            yaw = yaw  / M_PI * 180;
            tcpserver_->sendCameraPose(pos[0], pos[1], pos[2], 0 , yaw ,0);
            sleep(1 / send_freq_);
            if(i == 0){
                sleep(0.5);
                if(env_.find("gs") != std::string::npos){
                    tcpserver_->sendCameraPose(pos[0], pos[1], pos[2], 0 , yaw ,0);
                }
            }
        }
        saveActionsToJson(action_list, record_list,  pose_record_file_name_ + "pose.jsonl");
    }

    Object selectRandomObject(const std::vector<Object>& obj_list) {
        if (obj_list.empty()) {
            throw std::runtime_error("Object list is empty.");
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, obj_list.size() - 1);

        int index = dis(gen);
        return obj_list[index];
    }

    Object selectRandomObjectwithHeight(const std::vector<Object>& obj_list) {
        int trynums = 3000;
        if (obj_list.empty()) {
            throw std::runtime_error("Object list is empty.");
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, obj_list.size() - 1);
        int index = dis(gen);
        for(int i = 0; i < trynums; i ++){
            index = dis(gen);
            if( obj_list[index].pos.z() > flyheight_ - objinfov_height_thr_){
                return obj_list[index];
            }
        }
        return obj_list[index];
    }

    double selectRandomFlyheight(double h , double H){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> height_dist(h, H);
        return height_dist(gen);
    }

    Eigen::Vector3d selectRandomPointInShell(const Eigen::Vector3d& center, double r, double R, double h, double H) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> radius_dist(r, R);
        std::uniform_real_distribution<> height_dist(h, H);
        std::uniform_real_distribution<> angle_dist(0, 2 * M_PI); 

        double radius = radius_dist(gen);
        double height = height_dist(gen);
        double angle = angle_dist(gen);

        double x = center.x() + radius * cos(angle);
        double y = center.y() + radius * sin(angle);
        double z = height;

        return Eigen::Vector3d(x, y, z);
    }

    Eigen::Vector3d selectRandomPointInShell(const Eigen::Vector3d& center, double r, double R) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> radius_dist(r, R);
        std::uniform_real_distribution<> angle_dist(0, 2 * M_PI);

        double radius = radius_dist(gen);
        double angle = angle_dist(gen);

        double x = center.x() + radius * cos(angle);
        double y = center.y() + radius * sin(angle);
        double z = flyheight_;

        return Eigen::Vector3d(x, y, z);
    }


    Object selectRandomObjectInRange(const Eigen::Vector3d& center, double r, double R) {
        std::random_device rd;
        std::mt19937 gen(rd());
        Object nullobj;
        
        std::uniform_int_distribution<> dist(0, obj_list_.size() - 1);
        int trynum = 0;
        Object closest_obj = nullobj;
        double closest_distance = std::numeric_limits<double>::infinity();

        while (true) {
            const Object& obj = obj_list_[dist(gen)];
            double distance = (obj.pos - center).norm();
            if (distance >= r && distance <= R) {
                return obj; 
            }
            if (distance < closest_distance) {
                closest_distance = distance;
                closest_obj = obj;
            }
            
            trynum += 1;
            if (trynum >= 1000) {
                return closest_obj; 
            }
        }
    }
    
    void publishPointsInRViz(const std::vector<Eigen::Vector3d>& points, const std::string& frame_id, const std::string& ns)
        {
        visualization_msgs::msg::MarkerArray marker_array;
        std::vector<std::array<float, 4>> colors = {
            {1.0, 0.0, 0.0, 1.0}, 
            {0.0, 1.0, 0.0, 1.0}, 
            {0.0, 0.0, 1.0, 1.0}  
        };
        for (size_t i = 0; i < points.size(); ++i) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id; 
            marker.header.stamp = node_->get_clock()->now();
            marker.ns = ns;
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.scale.x = 2.0; 
            marker.scale.y = 2.0; 
            marker.scale.z = 2.0; 

            marker.color.r = colors[i % colors.size()][0];
            marker.color.g = colors[i % colors.size()][1];
            marker.color.b = colors[i % colors.size()][2];
            marker.color.a = colors[i % colors.size()][3]; 

            marker.pose.position.x = points[i].x();
            marker.pose.position.y = points[i].y();
            marker.pose.position.z = points[i].z();

            marker_array.markers.push_back(marker);
        }
        show_point_Pub_->publish(marker_array);
    }

    Eigen::Vector3d findAimingPoint(const Eigen::Vector3d& start_p, const Eigen::Vector3d& end_p, double d) {
        Eigen::Vector3d direction = (start_p - end_p).normalized();
        Eigen::Vector3d current_point = end_p;
        
        while ((current_point - start_p).norm() > d) {
            current_point += direction * d; 
            if (!bevQuery(current_point)) {
                return current_point; 
            }
        }
        return start_p;
    }


    std::vector<geometry_msgs::msg::Pose> readPosesFromFile(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<geometry_msgs::msg::Pose> poses;
        std::string line;

        while (std::getline(file, line)) {
            nlohmann::json j = nlohmann::json::parse(line);
            geometry_msgs::msg::Pose pose;
            pose.position.x = j["position"]["x"];
            pose.position.y = j["position"]["y"];
            pose.position.z = j["position"]["z"];
            pose.orientation.w = j["orientation"]["w"];
            pose.orientation.x = j["orientation"]["x"];
            pose.orientation.y = j["orientation"]["y"];
            pose.orientation.z = j["orientation"]["z"];

            poses.push_back(pose);
        }
        std::cout << "poses:" << poses.size() <<std::endl;
        return poses;
    }

    std::vector<Eigen::Vector3d> filterPointsinRange(const std::vector<Eigen::Vector3d>& point_cloud, double min_x, double max_x, double min_y, double max_y) {
        std::vector<Eigen::Vector3d> filtered_points;
        for (const auto& point : point_cloud) {
            double x = point[0];
            double y = point[1];
            if (x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
                filtered_points.push_back(point);
            }
        }
        return filtered_points;
    }

    bool isPShotP(Eigen::Vector3d start_p, Eigen::Vector3d end_p) {
        if (start_p == end_p) {
            std::cout << "The start point and end point are the same." << std::endl;
            return false;
        }
        Eigen::Vector3d direction = end_p - start_p;
        double distance = direction.norm(); 
        Eigen::Vector3d step = direction.normalized() * global_voxelmap_.getScale(); 
        Eigen::Vector3d current_point = start_p;
        while ((current_point - start_p).norm() < distance) {
            if (global_voxelmap_.query(current_point)) {
                return false;
            }
            current_point += step;
        }
        return true;

    }

    TrajRecodData genTrajRecordafterAstar(){
        return TrajRecodData(pathsearch_->getrecordlist(), 
                            pathsearch_->getactionlist() ,
                            cur_obj_);
    }

    bool isObjectInCloseList(const Object& obj, const std::vector<Object>& closeobj_list) {
        auto it = std::find(closeobj_list.begin(), closeobj_list.end(), obj);
        return it != closeobj_list.end();
    }

    void genTrajbyOneLandmark(int k, int r, int R, int h, int H, bool is_takeoff_landing, bool with_turn){
        // Select a random flight height within the range [h, H]
        flyheight_ = selectRandomFlyheight(map_height_thr_ + h, map_height_thr_ + H);
        // Generate k trajectories
        for(int i = 0; i < k; ) {
            std::cout << "Round: " << i << std::endl;  // Print current round number
            // Randomly select a target object and get its position
            Object target = selectRandomObjectwithHeight(obj_list_);
            cur_obj_ = target;  
            Eigen::Vector3d target_pos, end_point, start_point;
            target_pos = target.pos;  
            // Generate multiple candidate start points around the target object
            for(int j = 0; j < 20; j++) {
                // Randomly select a start point within a shell around the target (radius r to R)
                start_point = selectRandomPointInShell(target_pos, r, R);
                // Get the maximum Z value (height) at the start point, adjust based on the flight height
                double min_z = getMaxZinP(start_point[0], start_point[1], 2);
                if (flyheight_ < min_z) {
                    start_point.z() = min_z + 6;  // If flight height is below the minimum Z, adjust the start point height
                }
                // Update the target position's Z-coordinate to match the start point's height
                target_pos.z() = start_point.z();
                // Check if the start point is valid, continue if not
                if (bevQuery(start_point) == true or global_voxelmap_.query(start_point) == true) {
                    // std::cout << "Invalid start point, finding new start point"<< std::endl;
                    continue;  // If the start point is invalid, skip and try again
                }
                // Calculate the aiming point (target point) based on the start and target positions
                end_point = findAimingPoint(start_point, target_pos, 1.0);
        
                // If a turn is required, check if the path from start to end is valid
                if(with_turn) {
                    if(isPShotP(start_point, end_point)) {
                        // std::cout << "There are no obstacles between the start and end point" << std::endl;
                        continue;
                    } 
                }
                // Create a vector of points to represent the trajectory (target, end, and start points)
                std::vector<Eigen::Vector3d> points;
                points.push_back(target_pos);
                points.push_back(end_point);
                points.push_back(start_point);

                // Publish the points to RViz for visualization
                publishPointsInRViz(points, "map", "points_namespace");
                // Perform a path search from start to end point using A* algorithm
                astartSearchPath(start_point, end_point, true);
                // Record the trajectory path (including takeoff and landing if specified)
                recordPathbyTCP(is_takeoff_landing);
                i++;  // Increment trajectory counter
                break;  // Exit inner loop and try generating a new trajectory
            }
        }
    }

    void genTrajbyMultiLandmark(int k, int r, int R, int h, int H, int land_nums, bool with_turn, bool is_takeoff_landing){
        // Declare containers to store trajectory records and close objects encountered
        std::vector<TrajRecodData> traj_list;
        std::vector<Object> closeobj_list;
        double traj_yaw = 0;
        flyheight_ = selectRandomFlyheight(map_height_thr_ + h, map_height_thr_ + H);
         // Generate k trajectories
        for(int i = 0 ; i < k ;){
            closeobj_list.clear();
            traj_list.clear();
            std::cout << "Round: " << i << std::endl;
             // Select a random object as the starting target
            Object init_target = selectRandomObject(obj_list_);
            cur_obj_ = init_target;
            closeobj_list.push_back(cur_obj_);
            Eigen::Vector3d init_target_pos, init_end_point, init_start_point;
            init_target_pos = init_target.pos;
            bool init_flag = false;
            // std::cout <<"Target name:" << init_target.type << "  Target Position: " << init_target_pos.transpose() << std::endl;
            // Try to find a valid start and end point for the trajectory
            for(int j = 0 ; j < 20; j ++){
                init_start_point = selectRandomPointInShell(init_target_pos, r, R);
                double min_z = getMaxZinP(init_start_point[0], init_start_point[1], 2);
                if (flyheight_ < min_z) {
                    init_start_point.z() = min_z + 6;  // If flight height is below the minimum Z, adjust the start point height
                }
                init_target_pos.z() = init_start_point.z();
                if(bevQuery(init_start_point) == true or global_voxelmap_.query(init_start_point) == true){
                    // std::cout << "Invalid start point, finding new start point"<< std::endl;
                    continue;
                }
                init_end_point = findAimingPoint(init_start_point, init_target_pos, 1.0);
                if(with_turn){
                    if(isPShotP(init_start_point, init_end_point)){
                        // std::cout << "There are no obstacles between the start and end point" << std::endl;
                        continue;
                    } 
                }
                // std::cout << "init start point: " << init_start_point.transpose() <<  "init end point: " << init_end_point.transpose() << std::endl;
                astartSearchPath(init_start_point, init_end_point, true);

                traj_yaw = calculateYawDegree(init_end_point - init_start_point);
                TrajRecodData tmp_traj = genTrajRecordafterAstar();
                if(tmp_traj.que_size != 0){         
                    init_flag = true;
                    traj_list.push_back(tmp_traj);
                }
                break;
            }

            if(!init_flag) continue;
            Eigen::Vector3d target_pos, end_point, start_point;
            start_point = traj_list[0].record_list.back().head<3>();
            for(int land_i = 0 ; land_i < land_nums -1 ; land_i ++){
                for(int j = 0 ; j < 20 ; j ++){
                    cur_obj_ = selectRandomObjectInRange(start_point, r, R);
                    if(isObjectInCloseList(cur_obj_, closeobj_list )){
                        // std::cout << "Object already in close list, finding new object" << std::endl;
                        continue;
                    } 
                    closeobj_list.push_back(cur_obj_);
                    target_pos = cur_obj_.pos;
                    target_pos.z() = start_point.z();
                    end_point = findAimingPoint(start_point, target_pos, 1.0);
                    if(with_turn){
                        if(isPShotP(start_point, end_point)){
                            // std::cout << "There are no obstacles between the start and end point" << std::endl;
                            continue;
                        } 
                    }
                    // if(!isPShotP(start_point, end_point)) continue;
                    double new_yaw  = calculateYawDegree(end_point - start_point );
                    if(abs(new_yaw - traj_yaw) < 30 || abs(new_yaw - traj_yaw) > 150) {
                        // std::cout << "Yaw error too small, finding new object" << std::endl;
                        continue;
                    }
                    // std::cout << " start point: " << start_point.transpose() <<  " end point: " << end_point.transpose() << std::endl;
                    if( land_i == land_nums -2){
                        astartSearchPath(start_point, end_point, true);
                    }
                    astartSearchPath(start_point, end_point, true);
                    TrajRecodData tmp_traj = genTrajRecordafterAstar();
                    if(tmp_traj.que_size != 0){
                        traj_list.push_back(tmp_traj);
                    }
                    start_point = end_point;
                    break;
                }
            }
            if(traj_list.size() != land_nums) continue;
            recordPathbyTCP(traj_list, is_takeoff_landing);
            i++;
        }
    }

};


