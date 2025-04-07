#ifndef LOADPCDMAP
#define LOADPCDMAP

#include "rclcpp/rclcpp.hpp" 
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>     

#include <pcl/filters/statistical_outlier_removal.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

sensor_msgs::msg::PointCloud2 loadpcdfile(const std::string& pcdfilename) {
    // RCLCPP_INFO(rclcpp::get_logger("loadpcdfile"), "Loading PCD file, please wait...");
    
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    
    if (pcl::io::loadPCDFile(pcdfilename, *cloud) == -1) {
        RCLCPP_ERROR(rclcpp::get_logger("loadpcdfile"), "Failed to load PCD file: %s", pcdfilename.c_str());
        return cloud_msg;
    }
    
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";  
    // RCLCPP_INFO(rclcpp::get_logger("loadpcdfile"), "PCD file loaded successfully.");
    
    return cloud_msg;
}

sensor_msgs::msg::PointCloud2 filterPointCloud(const sensor_msgs::msg::PointCloud2& input_cloud, float leaf_size) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::fromROSMsg(input_cloud, *pcl_cloud);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(pcl_cloud);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*filtered_cloud);

    sensor_msgs::msg::PointCloud2 output_cloud;
    pcl::toROSMsg(*filtered_cloud, output_cloud);
    output_cloud.header = input_cloud.header; 

    return output_cloud;
}



sensor_msgs::msg::PointCloud2 removeOutliersWithStatisticalFilter(
    const sensor_msgs::msg::PointCloud2& input_cloud, int mean_k, double stddev_mul_thresh) {

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::fromROSMsg(input_cloud, *pcl_cloud);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(pcl_cloud);
    sor.setMeanK(mean_k);   
    sor.setStddevMulThresh(stddev_mul_thresh); 
    sor.filter(*filtered_cloud); 

    sensor_msgs::msg::PointCloud2 output_cloud;
    pcl::toROSMsg(*filtered_cloud, output_cloud);
    output_cloud.header = input_cloud.header;

    return output_cloud;
}


sensor_msgs::msg::PointCloud2 scalePointCloud(const sensor_msgs::msg::PointCloud2& input_cloud, double scale) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(input_cloud, *cloud);

    for (auto& point : cloud->points) {
        point.x *= scale;
        point.y *= scale;
        point.z *= scale;
    }

    sensor_msgs::msg::PointCloud2 output_cloud;
    pcl::toROSMsg(*cloud, output_cloud);
    output_cloud.header = input_cloud.header;

    return output_cloud;
}

sensor_msgs::msg::PointCloud2 loadPlyFile(const std::string& plyFilename) {
    RCLCPP_INFO(rclcpp::get_logger("loadPlyFile"), "Loading PLY file, please wait...");

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    std::ifstream file(plyFilename);
    if (!file.is_open()) {
        RCLCPP_ERROR(rclcpp::get_logger("loadPlyFile"), "Failed to open PLY file: %s", plyFilename.c_str());
        return cloud_msg;
    }

    std::string line;
    bool isHeader = true;
    size_t vertexCount = 0;

    while (std::getline(file, line) && isHeader) {
        if (line.find("end_header") != std::string::npos) {
            isHeader = false;
        } else if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string element;
            iss >> element >> element >> vertexCount;
        }
    }

    if (vertexCount == 0) {
        RCLCPP_ERROR(rclcpp::get_logger("loadPlyFile"), "No vertices found in PLY file.");
        return cloud_msg;
    }

    cloud->resize(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        if (!std::getline(file, line)) {
            RCLCPP_ERROR(rclcpp::get_logger("loadPlyFile"), "Failed to read vertex data.");
            return cloud_msg; 
        }
        std::istringstream iss(line);
        pcl::PointXYZ point;
        iss >> point.x >> point.y >> point.z; 
        cloud->points[i] = point;
    }

    file.close();
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.frame_id = "map";  
    RCLCPP_INFO(rclcpp::get_logger("loadPlyFile"), "PLY file loaded successfully.");
    
    return cloud_msg;
}


#endif
