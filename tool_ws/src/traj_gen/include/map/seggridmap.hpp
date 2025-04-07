#include <iostream>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "common.h"

class SegGridMap {

private:
    int map_width_;                
    int map_height_;               
    double grid_size_;             
    Eigen::Vector2d origin_;       
    std::vector<std::vector<std::vector<Object>>> grid_map_; 

public:
    SegGridMap() = default;
    SegGridMap(int width, int height, double grid_size, const Eigen::Vector2d& origin = Eigen::Vector2d(0.0, 0.0))
        : map_width_(width), map_height_(height), grid_size_(grid_size), origin_(origin) {
        grid_map_.resize(map_width_, std::vector<std::vector<Object>>(map_height_));
    }

    Eigen::Vector2i worldToGrid(const Eigen::Vector2d& world_pos) const {
        Eigen::Vector2d local_pos = world_pos - origin_; 
        int x = static_cast<int>(local_pos.x() / grid_size_);
        int y = static_cast<int>(local_pos.y() / grid_size_);
        return Eigen::Vector2i(x, y);
    }

    Eigen::Vector2d gridToWorld(const Eigen::Vector2i& grid_pos) const {
        double x = grid_pos.x() * grid_size_ + origin_.x();
        double y = grid_pos.y() * grid_size_ + origin_.y();
        return Eigen::Vector2d(x, y);
    }

    void addObjectToGrid(const Eigen::Vector2i& grid_pos, const Object& obj) {
        if (grid_pos.x() >= 0 && grid_pos.x() < map_width_ &&
            grid_pos.y() >= 0 && grid_pos.y() < map_height_) {
            grid_map_[grid_pos.x()][grid_pos.y()].push_back(obj);
        } else {
            std::cerr << "Grid position out of bounds!" << std::endl;
        }
    }

    void addObjectToGrid(const Eigen::Vector2d& world_pos, const Object& obj) {
        Eigen::Vector2i grid_pos = worldToGrid(world_pos);
        if (grid_pos.x() >= 0 && grid_pos.x() < map_width_ &&
            grid_pos.y() >= 0 && grid_pos.y() < map_height_) {
            grid_map_[grid_pos.x()][grid_pos.y()].push_back(obj);
        } else {
            std::cerr << "Grid position out of bounds!" << std::endl;
        }
    }

    void printGridInfo(const Eigen::Vector2i& grid_pos) const {
        if (grid_pos.x() >= 0 && grid_pos.x() < map_width_ &&
            grid_pos.y() >= 0 && grid_pos.y() < map_height_) {
            const auto& objects = grid_map_[grid_pos.x()][grid_pos.y()];
            std::cout << "Grid (" << grid_pos.x() << ", " << grid_pos.y() << ") contains "
                      << objects.size() << " object(s):" << std::endl;
            for (const auto& obj : objects) {
                std::cout << "Type: " << obj.type << ", Color: " << obj.color
                          << ", Size: " << obj.size << ", Shape: " << obj.shape
                          << ", Feature: " << obj.feature << ", Position: ("
                          << obj.pos.x() << ", " << obj.pos.y() << ", " << obj.pos.z()
                          << ")" << std::endl;
            }
        } else {
            std::cerr << "Grid position out of bounds!" << std::endl;
        }
    }

    std::vector<Object> queryObjectsByWorldCoord(const Eigen::Vector2d& world_pos) const {
        Eigen::Vector2i grid_pos = worldToGrid(world_pos);
        return queryObjectsByGridCoord(grid_pos);
    }

    std::vector<Object> queryObjectsByGridCoord(const Eigen::Vector2i& grid_pos) const {
        std::vector<Object> objects;
        if (grid_pos.x() >= 0 && grid_pos.x() < map_width_ && grid_pos.y() >= 0 && grid_pos.y() < map_height_) {
            objects = grid_map_[grid_pos.x()][grid_pos.y()];
        }
        return objects;
    }


    std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>> getVisibleObjects(
        const Eigen::Vector2d& position, double yaw_rad, double fov, double max_distance, int obj_num) {

        double half_fov = fov / 2.0;
        double left_angle = yaw_rad - half_fov * M_PI / 180.0;
        double right_angle = yaw_rad + half_fov * M_PI / 180.0;

        Eigen::Vector2i observer_grid = worldToGrid(position);
        std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>> visible_objects;

        int search_radius = std::ceil(max_distance / grid_size_); 

        for (int dx = -search_radius; dx <= search_radius; ++dx) {
            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                int target_x = observer_grid.x() + dx;
                int target_y = observer_grid.y() + dy;

                if (target_x < 0 || target_x >= map_width_ || target_y < 0 || target_y >= map_height_) {
                    continue;
                }
                Eigen::Vector2d grid_pos = gridToWorld(Eigen::Vector2i(target_x, target_y));
                double distance = (grid_pos - position).norm(); 

                if (distance > max_distance) continue;

                Eigen::Vector2d direction = grid_pos - position;
                double angle = std::atan2(direction.y(), direction.x());

                if (angle >= left_angle && angle <= right_angle) {
                    std::vector<Object> objects_in_grid = grid_map_[target_x][target_y];
                    if (!objects_in_grid.empty()) {
                        for(int i = 0; i < objects_in_grid.size(); i++){
                            objects_in_grid[i].addRelateinFov(position, fov, yaw_rad);
                        }
                        visible_objects.push_back(std::make_pair(gridToWorld(Eigen::Vector2i(target_x, target_y)), objects_in_grid));
                    }
                }
            }
        }
        std::sort(visible_objects.begin(), visible_objects.end(), 
            [&position](const std::pair<Eigen::Vector2d, std::vector<Object>>& a, 
                    const std::pair<Eigen::Vector2d, std::vector<Object>>& b) {
                double dist_a = (a.first - position).norm();
                double dist_b = (b.first - position).norm();
                return dist_a < dist_b;
            });
        if (visible_objects.size() > obj_num) {
            visible_objects.resize(obj_num);
        }
        return visible_objects;
    }

    std::vector<std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>>> getSeginFovbyRecod(
        std::vector<Eigen::Vector4d> record_list, double fov_in, double max_depth){
            std::vector<std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>>>  ans_segfov_list;
            std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>> tmp_objects;
            for(auto recod_p : record_list){
                Eigen::Vector2d tmp_pos(recod_p[0], recod_p[1]);
                double yaw(recod_p[3]);
                tmp_objects = getVisibleObjects(tmp_pos, yaw, fov_in, max_depth, 4);
                ans_segfov_list.push_back(tmp_objects);
            }
        return ans_segfov_list;
    }
    
};
