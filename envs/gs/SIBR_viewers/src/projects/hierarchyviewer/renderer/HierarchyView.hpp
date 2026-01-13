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


#pragma once


# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "common.h"
#include <types.h>
#include <chrono>
#include <future>

typedef Eigen::Matrix<float, 48, 1> SHs;

namespace sibr {

	class BufferCopyRenderer;

	/**
	 * \class RemotePointView
	 * \brief Wrap a ULR renderer with additional parameters and information.
	 */
	class SIBR_EXP_ULR_EXPORT HierarchyView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(HierarchyView);

	public:

		/**
		 * Constructor
		 * \param ibrScene The scene to use for rendering.
		 * \param render_w rendering width
		 * \param render_h rendering height
		 */
		HierarchyView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, const char* scaffoldfile, int64_t budget);

		/** Replace the current scene.
		 *\param newScene the new scene to render */
		void setScene(const sibr::BasicIBRScene::Ptr& newScene);

		/**
		 * Perform rendering. Called by the view manager or rendering mode.
		 * \param dst The destination rendertarget.
		 * \param eye The novel viewpoint.
		 */
		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;

		/**
		 * Update inputs (do nothing).
		 * \param input The inputs state.
		 */
		void onUpdate(Input& input) override;

		/**
		 * Update the GUI.
		 */
		void onGUI() override;

		/** \return a reference to the scene */
		const std::shared_ptr<sibr::BasicIBRScene>& getScene() const { return _scene; }

		virtual ~HierarchyView() override;

	protected:

		float biglimit = 30.0f;
		int cleanupFrequency = 100;

		float tau = 9.0f;
		float sizeLimit = 0.03f;

		int GAUSS_MEMLIMIT = 16000000;

		bool buffered = false;

		struct MemSet
		{
			Node* nodes_cuda;
			Box* boxes_cuda;
			sibr::Vector3f* pos_cuda;
			sibr::Vector4f* rot_cuda;
			sibr::Vector3f* scale_cuda;
			float* alpha_cuda;
			SHs* shs_cuda;
		};

		struct LightSet
		{
			int *to_render;
			int* active_nodes;
			int* nodes_of_render_indices;
			int* parent_indices;
			int* render_indices;
		};

		std::future< std::tuple<sibr::HierarchyView::MemSet*, int, int>> updateResult;

		MemSet mems[2];
		MemSet* currMem = nullptr;
		MemSet* otherMem = nullptr;

		LightSet lights[2];
		LightSet* currSet = nullptr;
		LightSet* otherSet = nullptr;

		bool ran_out = false;

		bool addNodePackage(const std::vector<int>& node_indices,
			const std::vector<int>& parent_indices,
			MemSet* useMem);

		int sibr::HierarchyView::createNodePackage(
			int num_to_expand,
			const int* indices_to_expand_cuda,
			std::vector<int>& node_indices,
			std::vector<int>& cuda_parent_indices,
			int* cuda_parent_starts
		);

		std::vector<sibr::Vector3f> pos;
		std::vector<sibr::Vector4f> rot;
		std::vector<SHs> shs;
		std::vector<float> alpha;
		std::vector<sibr::Vector3f> scale;

		Point* cam_pos;
		Point* cam_pos_old;

		int* package_parent_cuda_starts;
		int* need_children;

		int* new_node_count;
		int* new_gauss_count;

		int* newN, * newG, *newE;

		int* cuda2cpu;
		int* cuda2cpu1_cuda;
		int* cuda2cpu2_cuda;

		float usage_vals[100];
		int frame = 0;

		int* radii_cuda;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		bool showSfm = false;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* cam_pos_cuda_old;
		float* background_cuda;
		float* override_colors;

		float _scalingModifier = 1.0f;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;
		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;

		bool _useZFar = false;
		float _zfar = -1;

		std::shared_ptr<sibr::BasicIBRScene> _scene; ///< The current scene.
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;

		std::vector<int> activenodes1;
		std::vector<int> activenodes2;
		std::vector<int> render_indices;
		std::vector<int> splits;
		std::vector<Node> nodes;
		std::vector<Box> boxes;

		std::tuple<sibr::HierarchyView::MemSet*, int, int> asyncTask(Point* campos, Point zdir, bool cleanup);

		int* activenodes1_cuda;
		int* activenodes2_cuda;
		int* splits1_cuda;
		int* splits2_cuda;
		int* nodes_to_expand_cuda;
		int* rect_cuda;

		float* ts_cuda;
		int* kids_cuda;

		int cuda_nodes_offset = 0;
		int cuda_gaussians_offset = 0;

		int* num_active_nodes_gpu;
		int* num_need_children;
		int num_active_nodes_cpu;
		int* renderhelper;

		sibr::Matrix4f* view_mat_ptr;
		sibr::Matrix4f* proj_mat_ptr;

		Node* nodes_to_copy;
		Box* boxes_to_copy;
		sibr::Vector3f* pos_to_copy;
		sibr::Vector4f* rot_to_copy;
		SHs* shs_to_copy;
		float* alpha_to_copy;
		sibr::Vector3f* scale_to_copy;

		int skyboxnum = 0;

		bool disable_interp = false;
		bool show_level = false;
		bool m_use_cpu = false;

		cudaStream_t renderStream;
		cudaStream_t maintenanceStream;

		int* NsrcI, *NsrcI2;
		int* NdstI, * NdstI2;
		int* numI;
		char* NsrcC;
		char* scratchspace = nullptr;
		size_t scratchspacesize = 0;
	};

} /*namespace sibr*/
