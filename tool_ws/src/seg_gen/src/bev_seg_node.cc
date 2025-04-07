#include "bev_seg.cc"



int main(int argc, char** argv) {
    std::string env_name_ = "env_airsim_16";

    if(const char* env = std::getenv("ENV")) {
        env_name_ = (std::string)env;
        std::cout << "Your ENV is: " << env_name_ << '\n';
    }

    rclcpp::init(argc, argv);

    auto node = rclcpp::Node::make_shared("seg_gen_node");

    BEVMapGenerator bev_map_generator(node, env_name_);

    return 0;
}
