#ifndef OBJECT_HPP  
#define OBJECT_HPP 

#include <string>       
#include <Eigen/Dense>  
#include <nlohmann/json.hpp>


void normalYaw_degree(double& yaw){
    if(yaw > 180) yaw -= 360;
    if(yaw < -180) yaw += 360;
}
    double calculateYaw(const Eigen::Vector3d& vector) {
        double ans_yaw = std::atan2(vector.y(), vector.x());
        if (ans_yaw > M_PI){
            ans_yaw -= 2* M_PI;
        }
        if (ans_yaw <= -M_PI){
            ans_yaw += 2* M_PI;
        }
        return ans_yaw;
    }


    double calculateYawDegree(const Eigen::Vector3d& vector) {
        double ans_yaw = std::atan2(vector.y(), vector.x());
        if (ans_yaw > M_PI){
            ans_yaw -= 2* M_PI;
        }
        if (ans_yaw <= -M_PI){
            ans_yaw += 2* M_PI;
        }
        return ans_yaw * 180 / M_PI;
    }

struct Object {
    std::string type;
    std::string color;
    std::string size;
    std::string shape;
    std::string feature;
    Eigen::Vector3d pos;
    std::string relateinfov = "NONE";
    std::vector<Eigen::Vector3d> valid_points;

    Object() : type(""), color(""), size(""), shape(""), feature(""), pos(0.0, 0.0, 0.0) {}

    Object(const std::string& t, const std::string& c, const std::string& s,
            const std::string& sh, const std::string& f, const Eigen::Vector3d& p)
        : type(t), color(c), size(s), shape(sh), feature(f), pos(p) {}
    
    nlohmann::json toJson() const {
        nlohmann::json j;
        j["type"] = type;
        j["color"] = color;
        j["size"] = size;
        j["shape"] = shape;
        j["feature"] = feature;
        j["position"] = { pos.x(), pos.y(), pos.z() }; 
        if(relateinfov != "NONE"){ j["relateinfov"] = relateinfov;}
        return j;
    }

    void addRelateinFov(const Eigen::Vector2d& observer_pos, double fov_in, double in_yaw) {

        // std::cout << "in_ yaw" << in_yaw << std::endl;
        double yaw = in_yaw * 180.0 / M_PI;
        Eigen::Vector2d direction = Eigen::Vector2d(pos.x(), pos.y()) - Eigen::Vector2d(observer_pos.x(), observer_pos.y());
        double angle_to_object = std::atan2(direction.y(), direction.x()) * 180.0 / M_PI;  


        double yaw_left = yaw + fov_in / 6.0;  
        double yaw_right = yaw - fov_in / 6.0; 
        
        normalYaw_degree(yaw_left);
        normalYaw_degree(yaw_right);



        if (angle_to_object <= yaw_left && angle_to_object >= yaw_right) {
            relateinfov = "CENTER";
        } else if (angle_to_object > yaw_left) {
            relateinfov = "LEFT";   
        } else {
            relateinfov = "RIGHT";   
        }
    }



    bool operator==(const Object& other) const {
        return type == other.type && color == other.color && size == other.size && 
               shape == other.shape && feature == other.feature && pos == other.pos;
    }


};


struct TrajRecodData {

    int que_size;

    std::vector<Eigen::Vector4d> record_list;
    
    std::vector<std::pair<std::pair<std::string, int>, Eigen::Vector3d>> action_list;
    
    std::vector<std::vector<std::pair<Eigen::Vector2d, std::vector<Object>>>> seginfov_list;

    double start_yaw;

    double end_yaw;

    double traj_yaw;
    
    Object aim_obj;
    TrajRecodData(
        const std::vector<Eigen::Vector4d>& records,
        const std::vector<std::pair<std::pair<std::string, int>, Eigen::Vector3d>>& actions,
        Object obj
    ) : record_list(records), action_list(actions) ,aim_obj(obj){
        que_size = record_list.size();
    }
    TrajRecodData() = default;
    
};

#endif // OBJECT_HPP
