#include <iostream>
#include <thread>
#include <chrono>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <random>
#include <cstdlib>
#include <filesystem> 
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>


namespace fs = std::filesystem;

class BEVMapGenerator {
public:
    BEVMapGenerator(const rclcpp::Node::SharedPtr& node, const std::string& env_name)
    : node_(node), env_(env_name) {

        pr_dir_ = getProjectDir();
        
        yaml_path_ = pr_dir_ + "/configs/" + env_ + ".yaml";
        pcd_file_path_ = pr_dir_ + "/scene_data/pcd_map/" + env_ + ".pcd";
        seg_output_path_ = pr_dir_ + "/tool_ws/src/seg_gen/env_seg_info/" + env_;

        std::filesystem::path output_dir(seg_output_path_);
        if (!std::filesystem::exists(output_dir)) {
            bool success = std::filesystem::create_directories(output_dir); 
            if (!success) {
                std::cout<< "\033[31m" << "cannot create directory at " << seg_output_path_ << "\033[30m" <<std::endl;
                return;
            } 
        }

        loadYaml(yaml_path_);

        loadPCDAndGenerateBEV();
    }

private:
    std::shared_ptr<rclcpp::Node> node_; 
    std::string pcd_file_path_;
    double voxel_size_;
    std::string yaml_path_;
    std::string seg_output_path_;
    std::string pr_dir_;
    std::string pcd_map_path_;
    std::string output_path_;
    std::string seg_json_path_;
    std::string env_;
    double map_elevation_;
    double height_thresh_;
    double pcd_scale_ratio_;

    void loadYaml(std::string yaml_file){
        std::ifstream ifile(yaml_file);
        if (!ifile) {
            std::cerr << "YAML file not found: " << yaml_file << std::endl;
            return;
        }
            auto yaml = YAML::LoadFile(yaml_file);
            voxel_size_ = yaml["seg_map"]["bev_voxel_size"].as<double>();
            map_elevation_ = yaml["traj_map"]["map_elevation"].as<double>();
            height_thresh_ = yaml["traj_map"]["min_height_thresh"].as<double>();
            pcd_scale_ratio_ = yaml["traj_map"]["pcd_scale_ratio"].as<double>();

            std::cout << "voxel_size_" << voxel_size_ << std::endl;
    }


    std::string getProjectDir() {
        fs::path currentPath(__FILE__);
        fs::path pr_dir = currentPath.parent_path().parent_path().parent_path().parent_path().parent_path();
        return pr_dir.string();
    }


void loadPCDAndGenerateBEV() {

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path_, *cloud) == -1) {
        RCLCPP_ERROR(node_->get_logger(), " PCD file error: %s", pcd_file_path_.c_str());
        return;
    }
    RCLCPP_INFO(node_->get_logger(), "pcd load successful: %s", pcd_file_path_.c_str());
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr scaled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::cout << " pcd_scale_ratio_ " << pcd_scale_ratio_ << std::endl;
    for (const auto& point : *cloud) {
        scaled_cloud->points.push_back(pcl::PointXYZ(
            point.x * pcd_scale_ratio_,
            point.y * pcd_scale_ratio_,
            point.z * pcd_scale_ratio_
        ));
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::cout << "  map_elevation_ + height_thresh_ :" <<  map_elevation_ + height_thresh_ << std::endl;
    for (const auto& point : scaled_cloud->points) {
        if (point.z > map_elevation_ + height_thresh_) {
            filtered_cloud->points.push_back(point);
        }
    }


    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;
    std::cout << " point size :" << filtered_cloud->points.size() << std::endl;
    for (const auto& point : filtered_cloud->points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    // std::cout << " min_x" << min_x <<  "max_x" << max_x << std::endl;
    // std::cout << " voxel_size_" << voxel_size_ << std::endl;

    int map_width = static_cast<int>((max_x - min_x) / voxel_size_);
    int map_height = static_cast<int>((max_y - min_y) / voxel_size_);
    // std::cout << " map_width: " << map_width <<  "map_height" << map_height << std::endl;
    // std::cout <<" error before" << std::endl;

    cv::Mat bev_map(map_height, map_width, CV_8UC1, cv::Scalar(255));

    for (const auto& point : filtered_cloud->points) {

        int x = static_cast<int>((point.x - min_x) / voxel_size_);
        int y = static_cast<int>((point.y - min_y) / voxel_size_);


        if (x >= 0 && x < map_width && y >= 0 && y < map_height) {
            bev_map.at<uchar>(map_height - 1 - y, x) = 0; 
        }
    }


    std::string image_file_path = seg_output_path_ + "/" + env_ + ".pgm";
    cv::imwrite(image_file_path, bev_map);
    RCLCPP_INFO(node_->get_logger(), "BEV map saved %s", image_file_path.c_str());


    generateMapYaml(image_file_path, min_x , min_y);
}

    void generateMapYaml(const std::string& image_path, float min_x, float min_y) {
        std::ofstream yaml_file(seg_output_path_ + "/" + env_ + ".yaml");
        

        yaml_file << "image: " << image_path << "\n";
        yaml_file << "resolution: " << voxel_size_ << "\n";
        yaml_file << "origin: [" << min_x << ", " << min_y << ", 0.0]\n";  
        yaml_file.close();

        RCLCPP_INFO(node_->get_logger(), "map config file saved as %s/bev_map.yaml", seg_output_path_.c_str());
    }
};