#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <numeric> 
#include <Eigen/Eigen>
#include <rclcpp/rclcpp.hpp>

#include "map/voxel_map.hpp"
#include "common.h"

class PathSearch {
private:
    Eigen::Vector3d end_;
    voxel_map::VoxelMap global_map_;
    std::vector<Eigen::Vector4d> record_list_;
    std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> action_list_;  

    struct Node {
        Eigen::Vector3d pos;
        double yaw;
        double g_cost;
        double h_cost;
        std::shared_ptr<Node> parent;

        Node(Eigen::Vector3d pos_, double yaw_, double g, double h, std::shared_ptr<Node> p = nullptr)
            : pos(pos_), yaw(yaw_), g_cost(g), h_cost(h), parent(p) {}

        double f_cost() const {
            return g_cost + h_cost; // f(x) = g(x) + h(x)
        }
    };

    struct CompareNode {
        bool operator()(std::shared_ptr<Node> n1, std::shared_ptr<Node> n2) {
            return n1->f_cost() > n2->f_cost();  
        }
    };
    
    double cal_dis(std::shared_ptr<Node> node, Eigen::Vector3d end) {
        return (node->pos - end).norm();
    }


    double cal_dis(Node* node, Eigen::Vector3d end) {
        return (node->pos - end).norm();
    }

    double cal_dis(Eigen::Vector3d start, Eigen::Vector3d end) {
        return (start - end).norm();
    }

    double cal_dis(Node* node1, Node* node2) {
        return (node1->pos - node2->pos).norm();
    }

    double cal_dis(std::shared_ptr<Node> node1, std::shared_ptr<Node> node2) {
        return (node1->pos - node2->pos).norm();
    }

public:
    PathSearch(voxel_map::VoxelMap global_map, std::shared_ptr<rclcpp::Node> node)
        : global_map_(global_map) {
    }

    double Manhattan(Eigen::Vector3d v1, Eigen::Vector3d v2) {
        return (v1 - v2).cwiseAbs().sum();
    }

    // 计算启发函数：欧几里得距离
    double heuristic(Eigen::Vector3d v1, Eigen::Vector3d v2) {
        return (v1 - v2).norm();
    }

    bool isocc(Eigen::Vector3d pos) {
        return global_map_.query(pos);
    }

    std::vector<std::shared_ptr<Node>> getNeighbors(std::shared_ptr<Node> current) {
        std::vector<std::shared_ptr<Node>> neighbors;
        Eigen::Vector3d cur_node_pos = current->pos;
        double cur_yaw = current->yaw;

        std::vector<Eigen::Vector3d> motions;
        std::vector<double> angles = {0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150};
        int step_size = 3;

        for (const double angle : angles) {
            double rad = angle * M_PI / 180.0;
            motions.push_back(Eigen::Vector3d{step_size * cos(rad), step_size * sin(rad), 0});
        }
        motions.push_back(Eigen::Vector3d{0, 0, 3}); 
        motions.push_back(Eigen::Vector3d{0, 0, -3});

        for (int i = 0; i < motions.size(); ++i) {
            const auto& motion = motions[i];
            Eigen::Vector3d tmp_pos = cur_node_pos + motion;
            double angle_cost = (i < angles.size() && angles[i] != cur_yaw) ? 2.0 : 0;
            if (!isocc(tmp_pos)) {
                double new_yaw = angles[i];
                double tmp_gcost = heuristic(cur_node_pos, tmp_pos);
                neighbors.push_back(std::make_shared<Node>(tmp_pos, new_yaw, current->g_cost + tmp_gcost + angle_cost, heuristic(tmp_pos, end_), current));
            }
        }
        return neighbors;
    }


std::vector<Eigen::Vector3d> hybridAStar(Eigen::Vector3d start, Eigen::Vector3d end) {
    end_ = end;
    std::priority_queue<std::shared_ptr<Node>, std::vector<std::shared_ptr<Node>>, CompareNode> open_list;
    std::unordered_map<int, std::shared_ptr<Node>> closed_list;
    double thr = 3.1;

    std::shared_ptr<Node> start_node = std::make_shared<Node>(start, 0, 0, heuristic(start, end));
    open_list.push(start_node);

    if (isocc(start_node->pos)) {
        std::cerr << "Start point is occupied!" << std::endl;
        return {}; 
    }

    auto start_time = rclcpp::Clock().now();
    auto timeout_duration = std::chrono::minutes(2);  

    while (!open_list.empty() && rclcpp::ok()) {

        if (rclcpp::Clock().now() - start_time > timeout_duration) {
            std::cout << "\033[33m" << "Search timeout exceeded 2 minutes." << "\033[0m" << std::endl;
            return {}; 
        }


        std::shared_ptr<Node> current = open_list.top();
        open_list.pop();

        if (heuristic(current->pos, end) < thr) {
            std::vector<Eigen::Vector3d> path;
            while (current != nullptr) {
                path.push_back(current->pos);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        std::vector<std::shared_ptr<Node>> neighbors = getNeighbors(current);
        for (std::shared_ptr<Node> neighbor : neighbors) {
            int neighbor_hash = static_cast<int>(neighbor->pos[0]) * 1000000 + 
                                 static_cast<int>(neighbor->pos[1]) * 1000 + 
                                 static_cast<int>(neighbor->pos[2]);

            if (closed_list.find(neighbor_hash) != closed_list.end()) {
                continue;
            }

            neighbor->h_cost = heuristic(neighbor->pos, end);
            open_list.push(neighbor);
            closed_list[neighbor_hash] = neighbor;
        }
    }
    return {};
}

    std::vector<Eigen::Vector4d> caluavrecord(Eigen::Vector3d cur_p, Eigen::Vector3d next_p, double in_yaw, double out_yaw) {
        std::vector<Eigen::Vector4d> record_list;
        double yaw_error = std::round((out_yaw - in_yaw) / M_PI * 180);
        if(yaw_error < -180) yaw_error += 360;
        if(yaw_error > 180) yaw_error -= 360;


        int turn_nums = std::abs(std::round(yaw_error / 30.0));
        double step_yaw = (M_PI) / 180 * 30;
        if (yaw_error != 0) {
            if (yaw_error > 0) {
                for(int i = 0 ; i <= turn_nums; i ++){
                    record_list.emplace_back(cur_p[0], cur_p[1], cur_p[2], in_yaw + i * step_yaw);
                }

            } else {
                for(int i = 0 ; i <= turn_nums; i ++){
                    record_list.emplace_back(cur_p[0], cur_p[1], cur_p[2], in_yaw - i * step_yaw);
                }
            }
        }
        else{
            record_list.emplace_back(cur_p[0], cur_p[1], cur_p[2], in_yaw);
        }

        return record_list;
    }
    

    std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> caluavaction(Eigen::Vector3d start, Eigen::Vector3d end, double in_yaw, double out_yaw) {
        std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> action_list;
        double yaw_error = std::round((out_yaw - in_yaw) / M_PI * 180);  // yaw error in degrees
        
        if (yaw_error < -180) yaw_error += 360;  // Normalize to [-180, 180]
        if (yaw_error > 180) yaw_error -= 360;

        int turn_nums = std::abs(std::round(yaw_error / 30.0));  // Number of 30-degree turns required

        if(end[2] - start[2] > 1){
            action_list.emplace_back(std::make_pair("go up", std::round(cal_dis(start, end))), start);
            return action_list;  
        }
        if(end[2] - start[2] < -1){
            action_list.emplace_back(std::make_pair("go down", std::round(cal_dis(start, end))), start);  
            return action_list;
        }
        

        if (yaw_error != 0) {
            if (yaw_error > 0) {
                // std::cout << "turn_nums: " << turn_nums << std::endl;
                for (int i = 0; i < turn_nums; i++) {
                    action_list.emplace_back(std::make_pair("turn left", 30), start);  // Turning left
                }
            } else {
                // std::cout << "turn_nums: " << turn_nums << std::endl;
                for (int i = 0; i < turn_nums; i++) {
                    action_list.emplace_back(std::make_pair("turn right", 30), start);  // Turning right
                }
            }
            
            // Move straight after turning
            action_list.emplace_back(std::make_pair("go straight", std::round(cal_dis(start, end))), start);  
        } else {
            // If no yaw error, just go straight
            action_list.emplace_back(std::make_pair("go straight", std::round(cal_dis(start, end))), start);
        }

        return action_list;
    }




    void backtrackpath(const std::vector<Eigen::Vector3d>& path, bool with_stop) {
        record_list_.clear();
        action_list_.clear();
        if (path.size() <= 2) {
            // RCLCPP_WARN(node_->get_logger(), "Path error: Path size is too small.");
            return;
        }
        double start_yaw = calculateYaw(path[1] - path[0]);
        for (int i = 0; i < path.size() - 1; ++i) {
            double in_yaw = 0;
            double out_yaw = calculateYaw(path[i + 1] - path[i]);
            if(i == 0){
                in_yaw = start_yaw;
            }
            else{
                in_yaw = calculateYaw(path[i] - path[i - 1]);
                if(abs(path[i + 1][2] - path[i][2]) > 1){
                    in_yaw = record_list_[record_list_.size() -1][3];
                    out_yaw = in_yaw;
                }
                else if(abs(path[i][2] - path[i - 1][2]) > 1 ){
                    in_yaw = record_list_[record_list_.size() -1][3];
                    // out_yaw = in_yaw;
                }
            }


            auto tmp_act_list = caluavaction(path[i], path[i + 1], in_yaw, out_yaw);
            auto tmp_rec_list = caluavrecord(path[i], path[i + 1], in_yaw, out_yaw);

            action_list_.insert(action_list_.end(), tmp_act_list.begin(), tmp_act_list.end());
            record_list_.insert(record_list_.end(), tmp_rec_list.begin(), tmp_rec_list.end());
        }
        double end_yaw = record_list_.back()[3];
        if(with_stop){
            record_list_.emplace_back(path.back()[0], path.back()[1], path.back()[2], end_yaw);
            action_list_.emplace_back(std::make_pair("stop",0), path.back());
        }
    }

    std::vector<Eigen::Vector4d> getrecordlist() {
        return record_list_;
    }

     std::vector<std::pair<std::pair<std::string,int>, Eigen::Vector3d>> getactionlist() {
        return action_list_;
    }
};
