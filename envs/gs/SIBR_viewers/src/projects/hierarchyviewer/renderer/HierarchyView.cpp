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

#include <projects/hierarchyviewer/renderer/HierarchyView.hpp>
#include <core/graphics/GUI.hpp>
#include <thread>
#include <boost/asio.hpp>

#include <runtime_maintenance.h>
#include <runtime_switching.h>
#include <hierarchy_loader.h>
#include <cuda_rasterizer/rasterizer.h>

#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>

namespace sibr
{
	/** Copy the content of an input texture to another rendertarget or to the window.
	If you need a basic copy, prefer using blit.
	\sa sibr::blit
	\ingroup sibr_renderer
	*/
	class BufferCopyRenderer
	{

	public:

		/** Constructor. You can specify custom shaders, refer to noproj.vert and copy.frag for examples.
		\param vertFile pah to the vertex shader file
		\param fragFile pah to the fragment shader file
		*/
		BufferCopyRenderer()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("hierarchyviewer") + "/copy.vert"),
				sibr::loadFile(sibr::getShadersDirectory("hierarchyviewer") + "/copy.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		/** Copy input texture to the output texture, copy also the input alpha into depth.
		\param textureID the texture to copy
		\param dst the destination
		\param disableTest disable depth testing (depth won't be written)
		*/
		void	process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** Copy input texture to a window.
		\param textureID the texture to copy
		\param dst the destination window
		*/
		void	copyToWindow(uint textureID, Window& dst)
		{
			glDisable(GL_DEPTH_TEST);

			_shader.begin();

			glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textureID);
			sibr::RenderUtility::renderScreenQuad();

			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }

		int& width() { return _width.get(); }

		int& height() { return _height.get(); }

	private:

		GLShader			_shader; ///< Copy shader.
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};

	/** Copy the content of an input texture to another rendertarget or to the window.
	If you need a basic copy, prefer using blit.
	\sa sibr::blit
	\ingroup sibr_renderer
	*/
	class BufferCopyRenderer2
	{

	public:

		/** Constructor. You can specify custom shaders, refer to noproj.vert and copy.frag for examples.
		\param vertFile pah to the vertex shader file
		\param fragFile pah to the fragment shader file
		*/
		BufferCopyRenderer2()
		{
			_shader.init("CopyShader",
				sibr::loadFile(sibr::getShadersDirectory("hierarchyviewer") + "/copy2.vert"),
				sibr::loadFile(sibr::getShadersDirectory("hierarchyviewer") + "/copy2.frag"));

			_flip.init(_shader, "flip");
			_width.init(_shader, "width");
			_height.init(_shader, "height");
		}

		/** Copy input texture to the output texture, copy also the input alpha into depth.
		\param textureID the texture to copy
		\param dst the destination
		\param disableTest disable depth testing (depth won't be written)
		*/
		void	process(uint bufferID, IRenderTarget& dst, int width, int height, bool disableTest = true)
		{
			if (disableTest)
				glDisable(GL_DEPTH_TEST);
			else
				glEnable(GL_DEPTH_TEST);

			_shader.begin();
			_flip.send();
			_width.send();
			_height.send();

			dst.clear();
			dst.bind();

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID);

			sibr::RenderUtility::renderScreenQuad();

			dst.unbind();
			_shader.end();
		}

		/** Copy input texture to a window.
		\param textureID the texture to copy
		\param dst the destination window
		*/
		void	copyToWindow(uint textureID, Window& dst)
		{
			glDisable(GL_DEPTH_TEST);

			_shader.begin();

			glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, textureID);
			sibr::RenderUtility::renderScreenQuad();

			_shader.end();
		}

		/** \return option to flip the texture when copying. */
		bool& flip() { return _flip.get(); }

		int& width() { return _width.get(); }

		int& height() { return _height.get(); }

	private:

		GLShader			_shader; ///< Copy shader.
		GLuniform<bool>		_flip = false; ///< Flip the texture when copying.
		GLuniform<int>		_width = 1000;
		GLuniform<int>		_height = 800;
	};
}


typedef float SmallSHs[12];
typedef float Arr3[3];
typedef float Arr4[4];

struct RichPoint
{
	Arr3 pos;
	float n[3];
	SmallSHs shs;
	float alpha;
	Arr3 scale;
	Arr4 rot;
};

float sigmoidy(const float m1)
{
	return 1.0 / (1.0 + exp(-m1));
}

int loadScaffold(const char* path,
	std::vector<sibr::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<sibr::Vector3f>& scales,
	std::vector<sibr::Vector4f>& rot)
{
	std::string txtfile = (std::string(path) + "/pc_info.txt").c_str();
	std::string plyfile = (std::string(path) + "/point_cloud.ply").c_str();

	std::ifstream descfile(txtfile.c_str());
	std::string line;

	if (!descfile.good())
		throw std::runtime_error("Scaffold description not found! " + txtfile);

	std::getline(descfile, line);
	int count = std::atoi(line.c_str());

	std::ifstream infile(plyfile.c_str(), std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error("Scaffold not found! " + plyfile);

	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int noerp;
	ss >> dummy >> dummy >> noerp;

	while (std::getline(infile, buff))
	{
		if (buff.compare("end_header") == 0)
			break;
	}

	std::vector<RichPoint> points(count);

	infile.read((char*)points.data(), count * sizeof(RichPoint));

	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	alphas.resize(count);

	for (int k = 0; k < count; k++)
	{
		int i = k;
		pos[k] = { points[i].pos[0], points[i].pos[1], points[i].pos[2] };
		rot[k] = { points[i].rot[0], points[i].rot[1], points[i].rot[2], points[i].rot[3] };
		scales[k] = {
			expf(points[i].scale[0]),
			expf(points[i].scale[1]),
			expf(points[i].scale[2])
		};
		alphas[k] = sigmoidy(points[i].alpha);
		shs[k][0] = points[i].shs[0];
		shs[k][1] = points[i].shs[1];
		shs[k][2] = points[i].shs[2];
		for (int j = 1; j < 4; j++)
		{
			shs[k][(j - 1) + 3] = points[i].shs[j * 3 + 0];
			shs[k][(j - 1) + 18] = points[i].shs[j * 3 + 1];
			shs[k][(j - 1) + 33] = points[i].shs[j * 3 + 2];
		}
		for (int j = 4; j < 16; j++)
		{
			shs[k][(j - 1) + 3] = 0;
			shs[k][(j - 1) + 18] = 0;
			shs[k][(j - 1) + 33] = 0;
		}
	}

	return pos.size();
}

int loadHierarchy(const char* filename,
	std::vector<Eigen::Vector3f>& pos,
	std::vector<SHs>& shs,
	std::vector<float>& alphas,
	std::vector<Eigen::Vector3f>& scales,
	std::vector<Eigen::Vector4f>& rot,
	std::vector<Node>& nodes,
	std::vector<Box>& boxes)
{
	HierarchyLoader loader;
	loader.load(filename, pos, shs, alphas, scales, rot, nodes, boxes);

	int P = pos.size();
	for (int i = 0; i < P; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			scales[i][j] = exp(scales[i][j]);
		}
	}

	return P;
}

bool sibr::HierarchyView::addNodePackage(
	const std::vector<int>& node_indices,
	const std::vector<int>& cuda_parent_indices,
	MemSet* useMem
)
{
	int node_copy_count = node_indices.size();
	int gaussian_copy_count = 0;
	for (const int& id : node_indices)
	{
		Node node = nodes[id];
		gaussian_copy_count += node.count_leafs + node.count_merged;
	}

	if (node_copy_count + cuda_nodes_offset > GAUSS_MEMLIMIT ||
		gaussian_copy_count + cuda_gaussians_offset > GAUSS_MEMLIMIT)
	{
		//std::cout << "Out of mem!" << std::endl;
		//sizeLimit = std::max(sizeLimit, 0.00001f);
		if (tau == 0)
			tau = 1.0f;
		tau *= 1.05f;
		return false;
	}

	int copied_gaussians = 0;
	for (int i = 0; i < node_indices.size(); i++)
	{
		int id = node_indices[i];
		int parent = cuda_parent_indices[i];
		Node node = nodes[id];

		int count = node.count_leafs + node.count_merged;
		for (int j = 0; j < count; j++)
		{
			int src = node.start + j;
			int dst = copied_gaussians + j;
			pos_to_copy[dst] = pos[src];
			rot_to_copy[dst] = rot[src];
			shs_to_copy[dst] = shs[src];
			alpha_to_copy[dst] = alpha[src];
			scale_to_copy[dst] = scale[src];
		}

		node.start_children = -1;
		node.start = cuda_gaussians_offset + copied_gaussians;
		node.parent = parent;

		nodes_to_copy[i] = node;
		boxes_to_copy[i] = boxes[id];

		cuda2cpu[cuda_nodes_offset + i] = id;

		copied_gaussians += count;
	}

	cudaMemcpyAsync(useMem->pos_cuda + cuda_gaussians_offset, pos_to_copy, sizeof(sibr::Vector3f) * gaussian_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->rot_cuda + cuda_gaussians_offset, rot_to_copy, sizeof(sibr::Vector4f) * gaussian_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->shs_cuda + cuda_gaussians_offset, shs_to_copy, sizeof(SHs) * gaussian_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->alpha_cuda + cuda_gaussians_offset, alpha_to_copy, sizeof(float) * gaussian_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->scale_cuda + cuda_gaussians_offset, scale_to_copy, sizeof(sibr::Vector3f) * gaussian_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->nodes_cuda + cuda_nodes_offset, nodes_to_copy, sizeof(Node) * node_copy_count, cudaMemcpyHostToDevice, maintenanceStream);
	cudaMemcpyAsync(useMem->boxes_cuda + cuda_nodes_offset, boxes_to_copy, sizeof(Box) * node_copy_count, cudaMemcpyHostToDevice, maintenanceStream);

	cuda_gaussians_offset += gaussian_copy_count;
	cuda_nodes_offset += node_copy_count;

	return true;
}

float tau2Limit(float tau, float tanfovx, int width)
{
	if (tau == 0)
		return 0;

	return (2.f * (tau + 0.5f)) * tanfovx / (0.5f * width);
}

int sibr::HierarchyView::createNodePackage(
	int num_to_expand,
	const int* indices_to_expand_cuda,
	std::vector<int>& node_indices,
	std::vector<int>& cuda_parent_indices,
	int* cuda_parent_starts)
{
	cudaMemcpyAsync(need_children, nodes_to_expand_cuda, sizeof(int) * num_to_expand, cudaMemcpyDeviceToHost, maintenanceStream);
	cudaStreamSynchronize(maintenanceStream);

	int num_get_children = 0;
	int node_package_count = 0;
	for (int i = 0; i < num_to_expand; i++)
	{
		int cuda_id = need_children[i];
		int node_id = cuda2cpu[cuda_id];
		node_package_count += nodes[node_id].count_children;
		
		num_get_children++;
	}

	node_indices.resize(node_package_count);
	cuda_parent_indices.resize(node_package_count);

	int nodes_expanded = 0;
	for (int k = 0; k < num_get_children; k++)
	{
		int cuda_id = need_children[k];
		int node_id = cuda2cpu[cuda_id];
		Node node = nodes[node_id];
		for (int i = 0; i < node.count_children; i++)
		{
			node_indices[nodes_expanded + i] = node.start_children + i;
			cuda_parent_indices[nodes_expanded + i] = cuda_id;
		}
		cuda_parent_starts[k] = cuda_nodes_offset + nodes_expanded;
		nodes_expanded += node.count_children;
	}

	return num_get_children;
}

size_t totalAlloc = 0;
template<typename T>
void allocTracked(T** t, size_t Size)
{
	totalAlloc += Size;
	cudaMalloc(t, Size);
}

std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				cudaFree(*ptr);

			S = 1.15f * N;
			cudaMalloc(ptr, S);
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

#define CUDA_SAFE(A)\
A;\
if(cudaDeviceSynchronize())\
SIBR_ERR << "ERROR: " << cudaDeviceSynchronize();

int64_t basecost(int skyboxnum)
{
	return (3 + 48 + 1 + 3 + 4) * 4 * 2 * skyboxnum +
		(2 + 1) * 4 * skyboxnum +
		(16 + 16 + 3 + 3) * 4 +
		3 * 4 +
		4;
}

int64_t per_gauss_cost()
{
	return (3 + 48 + 1 + 3 + 4) * 4 * 2 +
		(7 + 8) * 4 * 2 +
		(1 + 1 + 1) * 4 * 2 +
		(1 + 1 + 1 + 1) * 4 +
		(1 + 1 + 1 + 1 + 1) * 4 +
		(2 + 1) * 4 +
		((1 + 1 + 1 + 1) * 4 + 1);
}

sibr::HierarchyView::HierarchyView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h, const char* file, const char* scaffoldfile, int64_t budget) :
	_scene(ibrScene),
	sibr::ViewBase(render_w, render_h)
{
	_pointbasedrenderer.reset(new PointBasedRenderer());
	_copyRenderer = new BufferCopyRenderer();
	_copyRenderer->flip() = true;
	_copyRenderer->width() = render_w;
	_copyRenderer->height() = render_h;


	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto& cams = ibrScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);

	std::vector<Eigen::Vector3f> eigenpos;
	std::vector<Eigen::Vector3f> eigenscale;
	std::vector<Eigen::Vector4f> eigenrot;

	loadHierarchy(file,
		eigenpos,
		shs,
		alpha,
		eigenscale,
		eigenrot,
		nodes,
		boxes);

	pos.resize(eigenpos.size());
	rot.resize(eigenpos.size());
	scale.resize(eigenpos.size());
	for (int i = 0; i < eigenpos.size(); i++)
	{
		pos[i] = eigenpos[i];
		rot[i] = eigenrot[i];
		scale[i] = eigenscale[i];
	}

	std::vector<sibr::Vector3f> skyboxpos;
	std::vector<sibr::Vector4f> skyboxrot;
	std::vector<SHs> skyboxsh;
	std::vector<float> skyboxalpha;
	std::vector<sibr::Vector3f> skyboxscale;

	if (strlen(scaffoldfile))
	{
		skyboxnum = loadScaffold(scaffoldfile,
			skyboxpos,
			skyboxsh,
			skyboxalpha,
			skyboxscale,
			skyboxrot);
	}

	GAUSS_MEMLIMIT = (budget*1000000 - basecost(skyboxnum)) / per_gauss_cost();
	if (GAUSS_MEMLIMIT < 0)
	{
		throw std::runtime_error("Memory budget insufficient");
	}

	GAUSS_MEMLIMIT = std::min(GAUSS_MEMLIMIT, std::max((int)pos.size(), (int)nodes.size()));

	SIBR_LOG << "Allowing up to " << GAUSS_MEMLIMIT << " Gaussians in VRAM" << std::endl;

	splits = std::vector<int>(GAUSS_MEMLIMIT, 0);

	int ALLGAUSS = (GAUSS_MEMLIMIT + skyboxnum);

	for (int i = 0; i < 2; i++)
	{
		sibr::Vector3f *allPos, *allScales;
		SHs* allSHs;
		float* allAlpha;
		sibr::Vector4f* allRot;
		CUDA_SAFE(allocTracked((void**)&allPos, sizeof(sibr::Vector3f) * ALLGAUSS));
		cudaMemcpy(allPos, skyboxpos.data(), sizeof(sibr::Vector3f) * skyboxnum, cudaMemcpyHostToDevice);
		CUDA_SAFE(allocTracked((void**)&allSHs, sizeof(SHs) * ALLGAUSS));
		cudaMemcpy(allSHs, skyboxsh.data(), sizeof(SHs) * skyboxnum, cudaMemcpyHostToDevice);
		CUDA_SAFE(allocTracked((void**)&allAlpha, sizeof(float) * ALLGAUSS));
		cudaMemcpy(allAlpha, skyboxalpha.data(), sizeof(float) * skyboxnum, cudaMemcpyHostToDevice);
		CUDA_SAFE(allocTracked((void**)&allScales, sizeof(sibr::Vector3f) * ALLGAUSS));
		cudaMemcpy(allScales, skyboxscale.data(), sizeof(sibr::Vector3f) * skyboxnum, cudaMemcpyHostToDevice);
		CUDA_SAFE(allocTracked((void**)&allRot, sizeof(sibr::Vector4f) * ALLGAUSS));
		cudaMemcpy(allRot, skyboxrot.data(), sizeof(sibr::Vector4f) * skyboxnum, cudaMemcpyHostToDevice);
		mems[i].pos_cuda = allPos + skyboxnum;
		mems[i].shs_cuda = allSHs + skyboxnum;
		mems[i].alpha_cuda = allAlpha + skyboxnum;
		mems[i].scale_cuda = allScales + skyboxnum;
		mems[i].rot_cuda = allRot + skyboxnum;

		CUDA_SAFE(allocTracked((void**)&mems[i].nodes_cuda, sizeof(Node) * GAUSS_MEMLIMIT));
		CUDA_SAFE(allocTracked((void**)&mems[i].boxes_cuda, sizeof(Box) * GAUSS_MEMLIMIT));
	}

	cudaHostAlloc((void**)&nodes_to_copy, sizeof(Node) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&boxes_to_copy, sizeof(Box) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&pos_to_copy, sizeof(sibr::Vector3f) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&rot_to_copy, sizeof(sibr::Vector4f) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&shs_to_copy, sizeof(SHs) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&alpha_to_copy, sizeof(float) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&scale_to_copy, sizeof(sibr::Vector3f) * GAUSS_MEMLIMIT, 0);

	cudaHostAlloc((void**)&cam_pos, sizeof(Point), 0);
	cudaHostAlloc((void**)&cam_pos_old, sizeof(Point), 0);

	cudaHostAlloc((void**)&new_node_count, sizeof(int), 0);
	cudaHostAlloc((void**)&new_gauss_count, sizeof(int), 0);

	cudaHostAlloc((void**)&newN, sizeof(int), 0);
	cudaHostAlloc((void**)&newG, sizeof(int), 0);
	cudaHostAlloc((void**)&newE, sizeof(int), 0);
	cudaHostAlloc((void**)&renderhelper, sizeof(int), 0);
	cudaHostAlloc((void**)&view_mat_ptr, sizeof(sibr::Matrix4f), 0);
	cudaHostAlloc((void**)&proj_mat_ptr, sizeof(sibr::Matrix4f), 0);

	cudaHostAlloc((void**)&cuda2cpu, sizeof(int) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&package_parent_cuda_starts, sizeof(int) * GAUSS_MEMLIMIT, 0);
	cudaHostAlloc((void**)&need_children, sizeof(int) * GAUSS_MEMLIMIT, 0);

	for (int i = 0; i < 2; i++)
	{
		cudaHostAlloc((void**)&lights[i].to_render, sizeof(int), 0);
		CUDA_SAFE(allocTracked((void**)&lights[i].render_indices, sizeof(int) * GAUSS_MEMLIMIT));
		CUDA_SAFE(allocTracked((void**)&lights[i].parent_indices, sizeof(int) * GAUSS_MEMLIMIT));
		CUDA_SAFE(allocTracked((void**)&lights[i].nodes_of_render_indices, sizeof(int) * GAUSS_MEMLIMIT));
	}

	cudaHostAlloc((void**)&num_active_nodes_gpu, sizeof(int), 0);
	cudaHostAlloc((void**)&num_need_children, sizeof(int), 0);

	CUDA_SAFE(allocTracked((void**)&splits1_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(cudaMemcpy(splits1_cuda, splits.data(), sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE(allocTracked((void**)&splits2_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(cudaMemcpy(splits2_cuda, splits.data(), sizeof(int), cudaMemcpyHostToDevice));
	CUDA_SAFE(allocTracked((void**)&activenodes1_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked((void**)&activenodes2_cuda, sizeof(int) * GAUSS_MEMLIMIT));

	currSet = &lights[0];
	otherSet = &lights[1];

	currMem = &mems[0];
	otherMem = &mems[1];

	activenodes1.resize(GAUSS_MEMLIMIT);
	activenodes2.resize(GAUSS_MEMLIMIT);
	render_indices.resize(GAUSS_MEMLIMIT);

	CUDA_SAFE(allocTracked(&cuda2cpu1_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked(&cuda2cpu2_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked((void**)&nodes_to_expand_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked((void**)&ts_cuda, sizeof(float) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked((void**)&kids_cuda, sizeof(int) * GAUSS_MEMLIMIT));
	CUDA_SAFE(allocTracked((void**)&rect_cuda, 2 * sizeof(int) * ALLGAUSS));
	CUDA_SAFE(allocTracked((void**)&radii_cuda, sizeof(int) * ALLGAUSS));

	activenodes1[0] = 0;
	num_active_nodes_cpu = 1;
	*num_active_nodes_gpu = 1;

	CUDA_SAFE(cudaMemcpy(activenodes1_cuda, activenodes1.data(), sizeof(int), cudaMemcpyHostToDevice));

	cudaStreamCreate(&renderStream);
	cudaStreamCreate(&maintenanceStream);

	addNodePackage({ 0 }, { -1 }, currMem);

	for (int i = 0; i < 100; i++)
		usage_vals[i] = 0;

	CUDA_SAFE(allocTracked((void**)&view_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE(allocTracked((void**)&proj_cuda, sizeof(sibr::Matrix4f)));
	CUDA_SAFE(allocTracked((void**)&cam_pos_cuda, 3 * sizeof(float)));
	CUDA_SAFE(allocTracked((void**)&cam_pos_cuda_old, 3 * sizeof(float)));

	CUDA_SAFE(glCreateBuffers(1, &imageBuffer));
	CUDA_SAFE(glNamedBufferStorage(imageBuffer, render_w * render_h * 3 * sizeof(float), nullptr, GL_DYNAMIC_STORAGE_BIT));
	CUDA_SAFE(cudaGraphicsGLRegisterBuffer(&imageBufferCuda, imageBuffer, cudaGraphicsRegisterFlagsWriteDiscard));

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	allocTracked((void**)&background_cuda, 3 * sizeof(float));
	float bg[3] = { 0.0f, 0.0f, 0.0f };
	cudaMemcpy(background_cuda, bg, 3 * sizeof(float), cudaMemcpyHostToDevice);

	allocTracked(&NsrcI, sizeof(int) * GAUSS_MEMLIMIT);
	allocTracked(&NdstI, sizeof(int) * GAUSS_MEMLIMIT);
	allocTracked(&NsrcI2, sizeof(int)* GAUSS_MEMLIMIT);
	allocTracked(&NdstI2, sizeof(int)* GAUSS_MEMLIMIT);
	allocTracked(&NsrcC, sizeof(char)* GAUSS_MEMLIMIT);
	allocTracked(&numI, sizeof(int));

	std::cout << "Using " << totalAlloc << " bytes for scene" << std::endl;
}

void sibr::HierarchyView::setScene(const sibr::BasicIBRScene::Ptr& newScene) {
	_scene = newScene;

	// Tell the scene we are a priori using all active cameras.
	std::vector<uint> imgs_ulr;
	const auto& cams = newScene->cameras()->inputCameras();
	for (size_t cid = 0; cid < cams.size(); ++cid) {
		if (cams[cid]->isActive()) {
			imgs_ulr.push_back(uint(cid));
		}
	}
	_scene->cameras()->debugFlagCameraAsUsed(imgs_ulr);
}

std::tuple<sibr::HierarchyView::MemSet*, int, int> sibr::HierarchyView::asyncTask(Point* campos, Point zdir, bool cleanup)
{
	cudaMemcpyAsync(cam_pos_cuda_old, campos, sizeof(Point), cudaMemcpyHostToDevice, maintenanceStream);

	MemSet* useMem = currMem;

	int add_success;

	Switching::changeToSizeStep(
		sizeLimit,
		*num_active_nodes_gpu,
		activenodes1_cuda,
		activenodes2_cuda,
		(int*)useMem->nodes_cuda,
		(float*)useMem->boxes_cuda,
		cam_pos_cuda_old,
		zdir.xyz[0], zdir.xyz[1], zdir.xyz[2],
		splits1_cuda,
		otherSet->render_indices,
		otherSet->parent_indices,
		otherSet->nodes_of_render_indices,
		nodes_to_expand_cuda,
		nullptr,
		scratchspace,
		scratchspacesize,
		NsrcI,
		NdstI,
		NsrcC,
		numI,
		GAUSS_MEMLIMIT,
		add_success,
		num_active_nodes_gpu,
		otherSet->to_render,
		num_need_children,
		maintenanceStream
	);

	cudaStreamSynchronize(maintenanceStream);

	bool success = add_success == 1;

	if (!success)
		throw std::runtime_error("Doing a step didn't work");

	std::swap(activenodes1_cuda, activenodes2_cuda);

	int num_get_children = 0;
	int num_transferred = 0;
	if (!cleanup && *num_need_children > 0)
	{
		if (!ran_out)
		{
			std::vector<int> package_indices;
			std::vector<int> package_parent_cuda_indices;

			int num_new_parents = createNodePackage(
				*num_need_children,
				nodes_to_expand_cuda,
				package_indices,
				package_parent_cuda_indices,
				package_parent_cuda_starts);

			if (addNodePackage(package_indices, package_parent_cuda_indices, useMem))
			{
				cudaMemcpyAsync(NsrcI, package_parent_cuda_starts, sizeof(int) * num_new_parents, cudaMemcpyHostToDevice, maintenanceStream);
				cudaStreamSynchronize(maintenanceStream);
				num_transferred = package_indices.size();
				num_get_children = num_new_parents;
			}
			else
			{
				ran_out = true;
			}
		}
	}

	if(cleanup || ran_out)
	{
		Maintenance::compactPart1(
			cuda_nodes_offset,
			*num_active_nodes_gpu,
			activenodes1_cuda,
			activenodes2_cuda,
			(int*)currMem->nodes_cuda,
			(float*)currMem->boxes_cuda,
			(float*)currMem->pos_cuda,
			(float*)currMem->rot_cuda,
			(float*)currMem->shs_cuda,
			currMem->alpha_cuda,
			(float*)currMem->scale_cuda,
			splits1_cuda,
			(int*)otherMem->nodes_cuda,
			(float*)otherMem->boxes_cuda,
			(float*)otherMem->pos_cuda,
			(float*)otherMem->rot_cuda,
			(float*)otherMem->shs_cuda,
			otherMem->alpha_cuda,
			(float*)otherMem->scale_cuda,
			splits2_cuda,
			cuda2cpu1_cuda,
			cuda2cpu2_cuda,
			NsrcI,
			NsrcI2,
			NdstI,
			NdstI2,
			scratchspace,
			scratchspacesize,
			maintenanceStream,
			new_node_count
		);

		if (cuda_nodes_offset - *new_node_count > 10000 || ran_out) // not gonna bother otherwise
		{
			cudaMemcpyAsync(cuda2cpu1_cuda, cuda2cpu, sizeof(int) * cuda_nodes_offset, cudaMemcpyHostToDevice, maintenanceStream);

			Maintenance::compactPart2(
				cuda_nodes_offset,
				*num_active_nodes_gpu,
				activenodes1_cuda,
				activenodes2_cuda,
				(int*)currMem->nodes_cuda,
				(float*)currMem->boxes_cuda,
				(float*)currMem->pos_cuda,
				(float*)currMem->rot_cuda,
				(float*)currMem->shs_cuda,
				currMem->alpha_cuda,
				(float*)currMem->scale_cuda,
				splits1_cuda,
				(int*)otherMem->nodes_cuda,
				(float*)otherMem->boxes_cuda,
				(float*)otherMem->pos_cuda,
				(float*)otherMem->rot_cuda,
				(float*)otherMem->shs_cuda,
				otherMem->alpha_cuda,
				(float*)otherMem->scale_cuda,
				splits2_cuda,
				cuda2cpu1_cuda,
				cuda2cpu2_cuda,
				NsrcI,
				NsrcI2,
				NdstI,
				NdstI2,
				scratchspace,
				scratchspacesize,
				maintenanceStream,
				new_gauss_count
			);
			cudaStreamSynchronize(maintenanceStream);
			cudaMemcpyAsync(cuda2cpu, cuda2cpu2_cuda, sizeof(int)* *num_active_nodes_gpu, cudaMemcpyDeviceToHost, maintenanceStream);
			cudaStreamSynchronize(maintenanceStream);

			std::swap(cuda2cpu1_cuda, cuda2cpu2_cuda);
			std::swap(activenodes1_cuda, activenodes2_cuda);
			std::swap(splits1_cuda, splits2_cuda);
			useMem = otherMem;

			cuda_nodes_offset = *new_node_count;
			cuda_gaussians_offset = *new_gauss_count;

			ran_out = false;

			Switching::changeToSizeStep(
				sizeLimit,
				*num_active_nodes_gpu,
				activenodes1_cuda,
				activenodes2_cuda,
				(int*)useMem->nodes_cuda,
				(float*)useMem->boxes_cuda,
				cam_pos_cuda_old,
				zdir.xyz[0], zdir.xyz[1], zdir.xyz[2],
				splits1_cuda,
				otherSet->render_indices,
				otherSet->parent_indices,
				otherSet->nodes_of_render_indices,
				nodes_to_expand_cuda,
				nullptr,
				scratchspace,
				scratchspacesize,
				NsrcI,
				NdstI,
				NsrcC,
				numI,
				GAUSS_MEMLIMIT,
				add_success,
				num_active_nodes_gpu,
				otherSet->to_render,
				num_need_children,
				maintenanceStream
			);

			cudaStreamSynchronize(maintenanceStream);

			bool success = add_success == 1;

			if (!success)
				throw std::runtime_error("Doing a step didn't work");

			std::swap(activenodes1_cuda, activenodes2_cuda);
		}
	}

	return std::make_tuple(useMem, num_get_children, num_transferred);
}

void sibr::HierarchyView::onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye)
{
	auto view_mat = eye.view();
	auto proj_mat = eye.viewproj();
	view_mat.row(1) *= -1;
	view_mat.row(2) *= -1;
	proj_mat.row(1) *= -1;

	auto t = view_mat.row(2).transpose();
	Point zdir = { t.x(), t.y(), t.z() };

	auto inv = view_mat.inverse();
	*cam_pos = { inv(0, 3), inv(1, 3), inv(2, 3) };

	float* image_cuda;
	size_t bytes;
	cudaGraphicsMapResources(1, &imageBufferCuda, renderStream);
	cudaGraphicsResourceGetMappedPointer((void**)&image_cuda, &bytes, imageBufferCuda);

	frame++;

	buffered |= frame % cleanupFrequency == 0;

	if (frame == 1 
	|| (frame % 2 == 0 && updateResult.wait_for(std::chrono::seconds(0)) == std::future_status::ready))
	{
		if (frame == 1)
		{
			updateResult = std::async(
				std::launch::async, &sibr::HierarchyView::asyncTask,
				this,
				cam_pos,
				zdir,
				false);
		}

		auto res = updateResult.get();

		cudaStreamSynchronize(renderStream);

		std::swap(currSet, otherSet);
		if (std::get<0>(res) == otherMem)
			std::swap(otherMem, currMem);

		int num_get_children = std::get<1>(res);
		if (num_get_children)
		{
			Maintenance::updateStarts(
				(int*)currMem->nodes_cuda,
				num_get_children,
				nodes_to_expand_cuda,
				NsrcI,
				renderStream);
			cudaStreamSynchronize(renderStream);
		}

		updateResult = std::async(
			std::launch::async, &sibr::HierarchyView::asyncTask,
			this,
			cam_pos,
			zdir,
			buffered);
		buffered = false;
	}

	float fovy = eye.fovy();
	float fovx = 2.0f * atan(tan(eye.fovy() * 0.5f) * eye.aspect());
	float tan_fovx = tan(fovx * 0.5f);
	float tan_fovy = tan(fovy * 0.5f);
	sizeLimit = tau2Limit(tau, tan_fovx, _resolution.x());

	if (showSfm)
	{
		_pointbasedrenderer->process(_scene->proxies()->proxy(), eye, dst);
	}
	else
	{
		*view_mat_ptr = view_mat;
		*proj_mat_ptr = proj_mat;

		int* parent_ptr = nullptr;
		float* ts_ptr = nullptr;
		int* kids_ptr = nullptr;
		if (!disable_interp)
		{
			parent_ptr = currSet->parent_indices;
			ts_ptr = ts_cuda;
			kids_ptr = kids_cuda;
		}

		Switching::getTsIndexed(
			*currSet->to_render,
			currSet->nodes_of_render_indices,
			sizeLimit,
			(int*)currMem->nodes_cuda,
			(float*)currMem->boxes_cuda,
			cam_pos->xyz[0], cam_pos->xyz[1], cam_pos->xyz[2],
			zdir.xyz[0], zdir.xyz[1], zdir.xyz[2],
			ts_cuda,
			kids_cuda,
			renderStream
		);

		CudaRasterizer::Rasterizer::forward(
			geomBufferFunc,
			binningBufferFunc,
			imgBufferFunc,
			*currSet->to_render + skyboxnum,
			3,
			16,
			background_cuda,
			_resolution.x(), _resolution.y(),
			currSet->render_indices,
			parent_ptr,
			ts_ptr,
			kids_ptr,
			(float*)currMem->pos_cuda,
			(float*)currMem->shs_cuda,
			nullptr,
			(float*)currMem->alpha_cuda,
			(float*)currMem->scale_cuda,
			_scalingModifier,
			(float*)currMem->rot_cuda,
			nullptr,
			(float*)view_mat_ptr,
			(float*)proj_mat_ptr,
			(float*)cam_pos,
			tan_fovx,
			tan_fovy,
			false,
			image_cuda,
			nullptr,
			radii_cuda,
			rect_cuda,
			nullptr,
			nullptr,
			false,
			skyboxnum,
			renderStream,
			renderhelper,
			biglimit,
			true
		);
	}

	cudaGraphicsUnmapResources(1, &imageBufferCuda, renderStream);
	if (!showSfm)
	{
		_copyRenderer->process(imageBuffer, dst, _resolution.x(), _resolution.y());
	}
}

void sibr::HierarchyView::onUpdate(Input& input)
{
}

void sibr::HierarchyView::onGUI()
{
	const std::string guiName = "3D Gaussians";
	if (ImGui::Begin(guiName.c_str())) {

		ImGui::Checkbox("Show SfM", &showSfm);
		ImGui::SliderFloat("Scaling Modifier", &_scalingModifier, 0.001f, 1.0f);
		ImGui::Checkbox("Use ZFar", &_useZFar);
		if (_useZFar)
		{
			ImGui::InputFloat("Zfar", &_zfar);
		}
		ImGui::InputFloat("Tau (size limit)", &tau);

		ImGui::InputInt("Cleanup Rate", &cleanupFrequency);
		cleanupFrequency = std::max(10, cleanupFrequency);

		ImGui::Checkbox("Show Level", &show_level);
		ImGui::Checkbox("Disable Interp", &disable_interp);

		for (int i = 0; i < 99; i++)
			usage_vals[i] = usage_vals[i + 1];
		usage_vals[99] = cuda_gaussians_offset;
		ImGui::PlotLines("Active Gauss", usage_vals, 100, 0, "", 0, GAUSS_MEMLIMIT, ImVec2(0, 80.f));

		ImGui::InputFloat("Biglimit", &biglimit);
	}
	ImGui::End();
}

sibr::HierarchyView::~HierarchyView()
{
}
