/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#include <fstream>

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/system/String.hpp>
#include "projects/hierarchyviewer/renderer/HierarchyView.hpp" 

#include <core/renderer/DepthRenderer.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/view/SceneDebugView.hpp>

// for http server
#include <thread>
#include <core/view/httplib.h>
#include <core/view/InteractiveCameraHandler.hpp>
#include <core/assets/CameraRecorder.hpp>

#define PROGRAM_NAME "sibr_3Dhierarchy"
using namespace sibr;

const char* usage = ""
"Usage: " PROGRAM_NAME " -path <dataset-path>"    	                                "\n"
;

cv::Mat renderImage(const std::string& cameraParams, const std::string& imageParams) {
    // 解析并使用cameraParams和imageParams，进行相应的渲染
    // 这里只是示例，你可以调用现有的3D渲染函数
    // TODO: 实现实际渲染逻辑

    // 示例：创建一个简单的RGB图像
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 0, 0));  // 创建一个红色图像
    cv::putText(img, "Rendering based on camera & image params", cv::Point(50, 250),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    return img;
}

int main(int ac, char** av) {

	// Parse Command-line Args
	CommandLineArgs::parseMainArgs(ac, av);
	GaussianAppArgs myArgs;
	myArgs.displayHelpIfRequired();

	//const bool doVSync = !myArgs.vsync;
	myArgs.vsync = false;
	// rendering size
	uint rendering_width = myArgs.rendering_size.get()[0];
	uint rendering_height = myArgs.rendering_size.get()[1];
	
	// window size
	uint win_width = rendering_width; // myArgs.win_width;
	uint win_height = rendering_height; // myArgs.win_height;

	const char* toload = myArgs.modelPath.get().c_str();
	const char* scaffold = myArgs.scaffoldPath.get().c_str();


	// std::cout << "Model path      : " << toload   << std::endl;
	// std::cout << "Scaffold path   : " << scaffold << std::endl;

	// // 如果还想看渲染/窗口大小等：
	// auto rend_size = myArgs.rendering_size.get();
	// std::cout << "Rendering size  : " << rend_size[0] << " x " << rend_size[1] << std::endl;
	// std::cout << "VSync           : " << (myArgs.vsync ? "ON" : "OFF") << std::endl;



	// Window setup
	sibr::Window		window(PROGRAM_NAME, sibr::Vector2i(50, 50), myArgs, getResourcesDirectory() + "/hierarchy/" + PROGRAM_NAME + ".ini");

	sibr::BasicIBRScene::SceneOptions opts;
	opts.cameras = true;
	opts.images = false;
	opts.mesh = true;
	opts.renderTargets = false;
	opts.texture = false;
	// std::cout << " myArgs : " << myArgs << std::endl;
	BasicIBRScene::Ptr		scene(new BasicIBRScene(myArgs, opts));

	// Setup the scene: load the proxy, create the texture arrays.
	const uint flags = SIBR_GPU_LINEAR_SAMPLING | SIBR_FLIP_TEXTURE;

	// Fix rendering aspect ratio if user provided rendering size
	uint scene_width = scene->cameras()->inputCameras()[0]->w();
	uint scene_height = scene->cameras()->inputCameras()[0]->h();
	float scene_aspect_ratio = scene_width * 1.0f / scene_height;
	float rendering_aspect_ratio = rendering_width * 1.0f / rendering_height;

	if ((rendering_width > 0) && !myArgs.force_aspect_ratio ) {
		if (abs(scene_aspect_ratio - rendering_aspect_ratio) > 0.001f) {
			if (scene_width > scene_height) {
				rendering_height = rendering_width / scene_aspect_ratio;
			}
			else {
				rendering_width = rendering_height * scene_aspect_ratio;
			}
		}
	}

	
	// check rendering size
	rendering_width = (rendering_width <= 0) ? scene->cameras()->inputCameras()[0]->w() : rendering_width;
	rendering_height = (rendering_height <= 0) ? scene->cameras()->inputCameras()[0]->h() : rendering_height;
	Vector2u usedResolution(rendering_width, rendering_height);
	std::cerr << " USED RES " << usedResolution << " scene w h " << scene_width << " : " << scene_height <<  
		 " NAME " << scene->cameras()->inputCameras()[0]->name() << std::endl;



	const unsigned int sceneResWidth = usedResolution.x();
	const unsigned int sceneResHeight = usedResolution.y();

	HierarchyView::Ptr	pointBasedView(new HierarchyView(scene, sceneResWidth, sceneResHeight, toload, scaffold, myArgs.budget.get()));

	// Raycaster.
	std::shared_ptr<sibr::Raycaster> raycaster = std::make_shared<sibr::Raycaster>();
	raycaster->init();
	raycaster->addMesh(scene->proxies()->proxy());

	// Camera handler for main view.
	sibr::InteractiveCameraHandler::Ptr generalCamera(new InteractiveCameraHandler());
	generalCamera->setup(scene->cameras()->inputCameras(), Viewport(0, 0, (float)usedResolution.x(), (float)usedResolution.y()), raycaster, { -1.0f,-1.0f });

	// Add views to mvm.
	MultiViewManager        multiViewManager(window, false);

	if (myArgs.rendering_mode == 1) 
		multiViewManager.renderingMode(IRenderingMode::Ptr(new StereoAnaglyphRdrMode()));
	
	multiViewManager.addIBRSubView("Point view", pointBasedView, usedResolution, ImGuiWindowFlags_ResizeFromAnySide);
	multiViewManager.addCameraForView("Point view", generalCamera);

	// Top view
	const std::shared_ptr<sibr::SceneDebugView> topView(new sibr::SceneDebugView(scene, generalCamera, myArgs,  myArgs.imagesPath.get().c_str()));
	multiViewManager.addSubView("Top view", topView, usedResolution);
	topView->active(false);

	CHECK_GL_ERROR;

	// save images
	generalCamera->getCameraRecorder().setViewPath(pointBasedView, myArgs.dataset_path.get());
	if (myArgs.pathFile.get() !=  "" ) {
		generalCamera->getCameraRecorder().loadPath(myArgs.pathFile.get(), usedResolution.x(), usedResolution.y());
		generalCamera->getCameraRecorder().recordOfflinePath(myArgs.outPath, multiViewManager.getIBRSubView("Point view"), "");
		if( !myArgs.noExit )
			exit(0);
	}


	// for http server
    httplib::Server svr;
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Hello, this is a 3D viewer HTTP server!", "text/plain");
    });
    svr.Post("/render", [&](const httplib::Request& req, httplib::Response& res) {
        // 检查是否提供了camera和image参数
        if (!req.has_param("camera") || !req.has_param("image")) {
            res.status = 400;  // 错误请求
            res.set_content("Missing camera or image parameter", "text/plain");
            return;
        }

        // 获取camera和image参数
        std::string cameraParams = req.get_param_value("camera");
        std::string imageParams = req.get_param_value("image");
		std::string pathParams = req.get_param_value("path");
		// std::cout << "cameraParams: " << cameraParams << std::endl;
		generalCamera->getCameraRecorder().loadColmapFromHttp(cameraParams, imageParams, pathParams);
		std::cout << " Image saved to ./ImageOutput successfully!" << std::endl;
        // // 调用渲染函数生成图像
        // cv::Mat renderedImage = renderImage(cameraParams, imageParams);

        // // 将图像保存为PNG格式并发送
        // std::vector<uchar> buf;
        // cv::imencode(".png", renderedImage, buf);

        // res.set_content(reinterpret_cast<const char*>(buf.data()), buf.size(), "image/png");
		res.set_content("Image has been saved to ./ImageOutput successfully!", "text/plain");
    });

    std::thread server_thread([&]() {
        std::cout << "Starting HTTP server on port 18080..." << std::endl;
        svr.listen("0.0.0.0", 18080); 
		
    });


	// Main looooooop.
	while (window.isOpened()) {

		sibr::Input::poll();
		window.makeContextCurrent();
		if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) {
			window.close();
		}

		multiViewManager.onUpdate(sibr::Input::global());
		multiViewManager.onRender(window);

		window.swapBuffer();
		CHECK_GL_ERROR;
	}

	// for http server
    svr.stop();
    server_thread.join();

	return EXIT_SUCCESS;
}
