#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <thread>
#include <sstream>
#include <rclcpp/rclcpp.hpp>

class TCPServer : public rclcpp::Node {
private:
    std::string ip_address;
    int send_port;
    int listen_port;

public:
    std::string msg_get;

    TCPServer(std::string ip, int send, int listen)
        : Node("tcp_server"), ip_address(ip), send_port(send), listen_port(listen) {
        // 创建一个线程用于监听接收数据
        std::thread(&TCPServer::receiveData, this).detach();
    }

    std::string getmsg() {
        return msg_get;
    }

    void sendString(const std::string& message) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create socket.");
            return;
        }

        // 设置服务器地址
        sockaddr_in serverAddress{};
        serverAddress.sin_family = AF_INET;
        serverAddress.sin_addr.s_addr = inet_addr(ip_address.c_str());
        serverAddress.sin_port = htons(send_port);

        // 尝试连接到服务器
        while (connect(sock, reinterpret_cast<struct sockaddr*>(&serverAddress), sizeof(serverAddress)) < 0 && rclcpp::ok()) {
            RCLCPP_WARN(this->get_logger(), "Failed to connect to server, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 发送数据
        if (send(sock, message.c_str(), message.size(), 0) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to send data.");
        } else {
            // RCLCPP_INFO(this->get_logger(), "Data sent: %s", message.c_str());
        }

        close(sock);
    }

    void sendCameraPose(float x, float y, float z, float pitch, float yaw, float roll) {
        std::stringstream pose_stream;
        pose_stream << x << "," << y << "," << z << "," << pitch << "," << yaw << "," << roll;
        std::string pose_str = pose_stream.str();

        sendString(pose_str);
        // RCLCPP_INFO(this->get_logger(), "Sent camera pose: %s", pose_str.c_str());
    }

    void receiveData() {
        int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create socket.");
            return;
        }

        sockaddr_in serverAddress{};
        serverAddress.sin_family = AF_INET;
        serverAddress.sin_addr.s_addr = INADDR_ANY;
        serverAddress.sin_port = htons(listen_port);

        if (bind(serverSocket, reinterpret_cast<struct sockaddr*>(&serverAddress), sizeof(serverAddress)) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to bind socket.");
            close(serverSocket);
            return;
        }

        listen(serverSocket, 1);

        while (rclcpp::ok()) {
            sockaddr_in clientAddress{};
            socklen_t clientAddressLength = sizeof(clientAddress);
            int clientSocket = accept(serverSocket, reinterpret_cast<struct sockaddr*>(&clientAddress), &clientAddressLength);
            if (clientSocket < 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to accept connection.");
                continue;
            }

            char buffer[1024];
            memset(buffer, 0, sizeof(buffer));
            ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
            if (bytesRead < 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to receive data.");
            } else {
                msg_get = buffer;
                RCLCPP_INFO(this->get_logger(), "Received data: %s", msg_get.c_str());
            }

            close(clientSocket);
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        close(serverSocket);
    }
};
