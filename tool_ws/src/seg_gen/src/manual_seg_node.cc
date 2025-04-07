#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp> 
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>
#include "loadpcdmap.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;



std::string getProjectDir() {
    fs::path currentPath(__FILE__);
    
    fs::path rosDir = currentPath.parent_path().parent_path().parent_path().parent_path().parent_path();
    
    return rosDir.string();  
}



class PointCloudVisualizer : public rclcpp::Node
{
public:
    PointCloudVisualizer(std::string env) : Node("point_cloud_visualizer"), env_(env)
    {
        goal_pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10, std::bind(&PointCloudVisualizer::pose_callback, this, std::placeholders::_1));

        global_map_Pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/seg_gen/show/global_map", 10);

        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/seg_gen/show/seg_point", 10);
        

        pro_path_ = getProjectDir();
        pcd_file_ = pro_path_ + "/scene_data/pcd_map/" + env_ + ".pcd";
        seg_file_ = pro_path_ + "/scene_data/seg_map/" + env_ + ".jsonl";

        sensor_msgs::msg::PointCloud2 cloud_msg = loadpcdfile(pcd_file_);
        cloud_msg.header.frame_id = "map";
        cloud_msg.header.stamp = this->get_clock()->now();

        cloud_msg = filterPointCloud(cloud_msg, 2.0);
        cloud_msg = removeOutliersWithStatisticalFilter(cloud_msg, 60, 2.0);

        rclcpp::sleep_for(std::chrono::seconds(10));
        global_map_Pub_->publish(cloud_msg);

        jsonl_file_.open(seg_file_, std::ios::out | std::ios::app);
        if (!jsonl_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Unable to open JSONL file for writing.");
        }
    }

    ~PointCloudVisualizer()
    {
        if (jsonl_file_.is_open()) {
            jsonl_file_.close();
        }
    }

private:
    void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {

        nlohmann::ordered_json pose_data;
        pose_data["type"] = "building";
        pose_data["color"] = "test";
        pose_data["size"] = "test";
        pose_data["shape"] = "test";
        pose_data["feature"] = "test";


        std::ostringstream filename_stream;
        filename_stream << "X=" << std::setprecision(15) << msg->pose.position.x
                        << "Y=" << std::setprecision(15) << msg->pose.position.y
                        << "Z=" << std::setprecision(15) << msg->pose.position.z
                        << ".png";
        pose_data["filename"] = filename_stream.str();

        std::ofstream output_file(seg_file_, std::ios_base::app);
        output_file << pose_data.dump() << std::endl;
        frame_id++;

        publish_marker(msg);
    }

    void publish_marker(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "goal_pose";
        marker.id = frame_id;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose = msg->pose;
        marker.pose.position.z = 100;  
        marker.scale.x = 5; 
        marker.scale.y = 5;
        marker.scale.z = 5 ;
        marker.color.a = 1.0;  
        marker.color.r = 1.0; 
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        marker_pub_->publish(marker);
    }

    sensor_msgs::msg::PointCloud2 loadpcdfile(const std::string& pcdfilename)
    {
        RCLCPP_INFO(rclcpp::get_logger("loadpcdfile"), "Loading PCD file, please wait...");

        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

        if (pcl::io::loadPCDFile(pcdfilename, *cloud) == -1) {
            RCLCPP_ERROR(rclcpp::get_logger("loadpcdfile"), "Failed to load PCD file: %s", pcdfilename.c_str());
            return cloud_msg;
        }

        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "map";
        RCLCPP_INFO(rclcpp::get_logger("loadpcdfile"), "PCD file loaded successfully.");

        return cloud_msg;
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_Pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_; 
    std::ofstream jsonl_file_;
    std::string pro_path_;
    std::string pcd_file_;
    std::string seg_file_; 
    std::string env_;
    int frame_id = 0; 
};

int main(int argc, char **argv)
{


    std::string env_name_ = "env_airsim_16";

    if(const char* env = std::getenv("ENV")) {
        env_name_ = (std::string)env;
        std::cout << "Your ENV is: " << env_name_ << '\n';
    }
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PointCloudVisualizer>(env_name_);

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}
