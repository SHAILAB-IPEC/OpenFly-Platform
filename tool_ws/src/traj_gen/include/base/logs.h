/**
 * @file logs.h
 * @author WangLiansheng (wangliansheng@pjlab.org.cn)
 * @brief 
 * @version 0.1
 * @date 2023-05-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef CUSTOM_LOG_H_
#define CUSTOM_LOG_H_


#include <ctime>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <ros/package.h>

#include "base/parameter.h"

using namespace std;
using namespace google;

#define LOG_COLOR_RESET "\033[0m"
#define LOG_COLOR_RED "\033[31m"
#define LOG_COLOR_GREEN "\033[32m"
#define LOG_COLOR_YELLOW "\033[33m"
#define LOG_COLOR_BLUE "\033[34m"
#define LOG_COLOR_MAGENTA "\033[35m"
#define LOG_COLOR_CYAN "\033[36m"
#define LOG_COLOR_WHITE "\033[37m"


#define LOG_INFO(msg) LOG(INFO) << LOG_COLOR_GREEN << " ---> "<< msg << LOG_COLOR_RESET
#define LOG_INFO_R(msg) LOG(INFO) << LOG_COLOR_RED << " ---> "<< msg << LOG_COLOR_RESET
#define LOG_WARNING(msg) LOG(WARNING) << LOG_COLOR_YELLOW << " ---> "<< msg << LOG_COLOR_RESET
#define LOG_ERROR(msg) LOG(ERROR) << LOG_COLOR_RED << " ---> "<< msg << LOG_COLOR_RESET


void init_log(){
	google::InitGoogleLogging("Logs");
  FLAGS_logbufsecs = 0;
	FLAGS_alsologtostderr = true;
	FLAGS_minloglevel = google::INFO;
}

void save_log(const string& dir){
  auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  char time_str[100];
  strftime(time_str, sizeof(time_str), "%Y%m%d-%H%M%S", localtime(&now_c));
	std::string package_path, log_folder;
	if (dir == "local"){
		package_path = std::string(g_root_dir);
		log_folder = package_path + "/logs/" + time_str + "/";
		if (!std::filesystem::exists(log_folder)) {
			std::filesystem::create_directory(log_folder);
		}
	}else{
		log_folder = dir + "/" + time_str + "/";
		if (!std::filesystem::exists(log_folder)) {
			std::filesystem::create_directory(log_folder);
		}
	}
  FLAGS_log_dir = log_folder;
}

#endif