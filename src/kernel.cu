#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

#include "svd3.h" // credits to author

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


void checkCUDAError(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 512

/*! Size of the starting area in simulation space. */
#define scene_scale 0.050f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec3 *dev_pos;
glm::vec3 *dev_col;
glm::vec3 *Ybuffer_corr;

glm::vec3 *dev_Ybuffer;
glm::vec3 *dev_Xbuffer;
glm::vec3 *dev_YbufferCorr;
glm::mat3 *dev_intermMats;
glm::vec3 *dev_Ybuffer_m;
glm::vec3 *dev_Xbuffer_m;

cudaEvent_t start, stop;


/******************
* initCpuICP *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
}


__global__ void kernResetVec3Buffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

void Points::initCpuICP(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer) {

	int Y = Ybuffer.size();
	int X = Xbuffer.size();

	numObjects = Y + X;

	// Only for visulization

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_col, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

	cudaMemcpy(dev_pos, &Ybuffer[0], Ybuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_pos[Ybuffer.size()], &Xbuffer[0], Xbuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	kernResetVec3Buffer << <dim3((Y + blockSize - 1) / blockSize), blockSize >> > (Y, dev_col, glm::vec3(1, 1, 1));
	kernResetVec3Buffer << <dim3((X + blockSize - 1) / blockSize), blockSize >> > (X, &dev_col[Y], glm::vec3(1, 0, 0));

	cudaThreadSynchronize();
}


/******************
* copyPointsToVBO *
******************/

/**
* Copy the point positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	float c_scale = -1.0f / s_scale;

	if (index < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index < N) {
		vbo[4 * index + 0] = vel[index].x + 0.3f;
		vbo[4 * index + 1] = vel[index].y + 0.3f;
		vbo[4 * index + 2] = vel[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopypointsToVBO CUDA kernel.
*/
void Points::copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
	
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, vbodptr_positions, scene_scale);
	kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_col, vbodptr_velocities, scene_scale);

	checkCUDAErrorWithLine("copyPointsToVBO failed!");

	cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

__global__ void kernUpdatePos(int N, glm::vec3 *dev_pos, glm::vec3 val) {
	
	// Update position by velocity
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	//printf("index =  %d \n",index);
	if (index >= N) {
		return;
	}
	dev_pos[index] = dev_pos[index] + val;

}

__global__ void kernUpdateCol(int N, glm::vec3 *dev_col, int reducedSize) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}
	if (index >= reducedSize) {
		dev_col[index] = glm::vec3(0.0f, 0.0f, 0.0f);
	}
}


void findCorrespondencesCPU(int Y, int X, std::vector<glm::vec3>& Ybuffer_corr, std::vector<glm::vec3>& Ybuffer, std::vector<glm::vec3>& Xbuffer) {
	
	// size of Y_buffer corr is size of X

	for (int i = 0; i < X; i++) { // for each X

		Ybuffer_corr[i] = glm::vec3(0.0f,0.0f,0.0f);
		float dist = glm::distance(Ybuffer[0],Xbuffer[i]);

		for (int j = 1; j < Y; j++) { // all Y's

			float d = glm::distance(Ybuffer[j], Xbuffer[i]);
			
			if (dist > d) {
				dist = d;
				Ybuffer_corr[i] = Ybuffer[j];
			}
		}
		//std::cout<< std::endl << Xbuffer[i].x << " " << Xbuffer[i].y << " " << Xbuffer[i].z << std::endl;
		//std::cout << Ybuffer_corr[i].x << " " << Ybuffer_corr[i].y << " " << Ybuffer_corr[i].z << std::endl;
	}

}

glm::vec3 computeMeanCPU(int size, std::vector<glm::vec3>& Ybuffer_corr) {
	
	glm::vec3 mean(0.0f,0.0f,0.0f);
	
	for (int i = 0; i < size; i++) {
		mean = mean + Ybuffer_corr[i];
	}

	mean.x = mean.x / (1.0f * size);
	mean.y = mean.y / (1.0f * size);
	mean.z = mean.z / (1.0f * size);

	return mean;
}

void vec3AddCPU(int size, glm::vec3 addConst, std::vector<glm::vec3>& old_buff, std::vector<glm::vec3>& new_buff) {
	for (int i = 0; i < size; i++) {
		new_buff[i] =  old_buff[i] + addConst;
	}
}

void matmulCPU(int size, std::vector<glm::vec3>& Ybuffer_mean, std::vector<glm::vec3>& Xbuffer_mean, std::vector<glm::vec3>& Res) {
	
	// Dedicated matmul for 3xN - nx3 ==> 3x3
	// Y - Nx3
	// X - Nx3
	// compute - Yt*X
	
	for (int i = 0; i < size; i++) {
		
		Res[0].x = Res[0].x + (Ybuffer_mean[i].x * Xbuffer_mean[i].x);
		Res[0].y = Res[0].y + (Ybuffer_mean[i].x * Xbuffer_mean[i].y);
		Res[0].z = Res[0].z + (Ybuffer_mean[i].x * Xbuffer_mean[i].z);

		Res[1].x = Res[1].x + (Ybuffer_mean[i].y * Xbuffer_mean[i].x);
		Res[1].y = Res[1].y + (Ybuffer_mean[i].y * Xbuffer_mean[i].y);
		Res[1].z = Res[1].z + (Ybuffer_mean[i].y * Xbuffer_mean[i].z);

		Res[2].x = Res[2].x + (Ybuffer_mean[i].z * Xbuffer_mean[i].x);
		Res[2].y = Res[2].y + (Ybuffer_mean[i].z * Xbuffer_mean[i].y);
		Res[2].z = Res[2].z + (Ybuffer_mean[i].z * Xbuffer_mean[i].z);
	}
}

void matmulUVCPU(int size, std::vector<glm::vec3>& U, std::vector<glm::vec3>& V, std::vector<glm::vec3>& Rot) {

		Rot[0].x = glm::dot(U[0],V[0]);  
		Rot[0].y = glm::dot(U[0], V[1]); 
		Rot[0].z = glm::dot(U[0], V[2]); 

		Rot[1].x = glm::dot(U[1], V[0]); 
		Rot[1].y = glm::dot(U[1], V[1]); 
		Rot[1].z = glm::dot(U[1], V[2]); 

		Rot[2].x = glm::dot(U[2], V[0]); 
		Rot[2].y = glm::dot(U[2], V[1]); 
		Rot[2].z = glm::dot(U[2], V[2]);  

}

void computeTransCPU(int size, glm::vec3 mean_x, glm::vec3 mean_y, std::vector< glm::vec3>& Rot, glm::vec3& Trans) {
	
	Trans.x = mean_y.x - glm::dot(Rot[0],mean_x); 
	Trans.y = mean_y.y - glm::dot(Rot[1], mean_x);
	Trans.z = mean_y.z - glm::dot(Rot[2], mean_x); 

}

void updateXCPU(std::vector< glm::vec3>& Rot, glm::vec3& Trans, std::vector< glm::vec3>& Xbuffer, int size) {

	for (int i = 0; i < size; i++) {

		Xbuffer[i] = Trans + glm::vec3(glm::dot(Rot[0], Xbuffer[i]), 
									glm::dot(Rot[1], Xbuffer[i]), 
									glm::dot(Rot[2], Xbuffer[i]));
	}

}

void print3x3matrix(std::vector< glm::vec3>& Rot) {

	std::cout << Rot[0].x << " " << Rot[0].y << " " << Rot[0].z << " " << std::endl;
	std::cout << Rot[1].x << " " << Rot[1].y << " " << Rot[1].z << " " << std::endl;
	std::cout << Rot[2].x << " " << Rot[2].y << " " << Rot[2].z << " " << std::endl;

}

void Points::stepSimulationICPNaive(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer, float dt) {

	// Performace Measurment
	//cudaEventRecord(start);
	
	int Y = Ybuffer.size();
	int X = Xbuffer.size();

	numObjects = Y + X;

	std::cout << "======= dt:" << dt << std::endl;
	std::cout << "Finding Correspondences:" << std::endl;

	std::vector<glm::vec3> Ybuffer_corr(X);
	findCorrespondencesCPU(Y, X, Ybuffer_corr, Ybuffer, Xbuffer);

	std::cout << " Y coor 0 :" << Ybuffer_corr[0].x << " " << Ybuffer_corr[0].y << " " << Ybuffer_corr[0].z << std::endl;
	std::cout << " Y corr X :" << Ybuffer_corr[X-1].x << " " << Ybuffer_corr[X-1].y << " " << Ybuffer_corr[X-1].z << std::endl;
	
	// Compute Mean
	std::vector<glm::vec3> Ybuffer_mean(X);
	std::vector<glm::vec3> Xbuffer_mean(X);
	
	glm::vec3 mean_y = computeMeanCPU(X, Ybuffer_corr);
	glm::vec3 mean_x = computeMeanCPU(X, Xbuffer);

	std::cout << " X mean :"<< mean_x.x << " " << mean_x.y << " " << mean_x.z << std::endl;
	std::cout << " Y mean :"<< mean_y.x << " " << mean_y.y << " " << mean_y.z << std::endl;

	// Mean Center
	vec3AddCPU(X, -1.0f * mean_x, Xbuffer, Xbuffer_mean);
	vec3AddCPU(X, -1.0f * mean_y, Ybuffer_corr, Ybuffer_mean);

	// Compute Yt*X 
	std::vector<glm::vec3> YtX(3);
	matmulCPU(X, Ybuffer_mean, Xbuffer_mean, YtX);
	
	std::cout << "YtX Matrix \n";
	print3x3matrix(YtX);


	// compute SVD of Yt X
	std::vector<glm::vec3> Rot(3, glm::vec3(0.0f, 0.0f, 0.0f));
	glm::vec3 Trans;
	
	std::vector<glm::vec3> U(3, glm::vec3(0.0f, 0.0f, 0.0f));
	std::vector<glm::vec3> S(3,glm::vec3(0.0f,0.0f,0.0f));
	std::vector<glm::vec3> V(3, glm::vec3(0.0f, 0.0f, 0.0f));

	// compute SVD
	svd(YtX[0].x, YtX[0].y, YtX[0].z,
		YtX[1].x, YtX[1].y, YtX[1].z,
		YtX[2].x, YtX[2].y, YtX[2].z,
		
		U[0].x, U[0].y, U[0].z,
		U[1].x, U[1].y, U[1].z,
		U[2].x, U[2].y, U[2].z,
		
		S[0].x, S[0].y, S[0].z,
		S[1].x, S[1].y, S[1].z,
		S[2].x, S[2].y, S[2].z,
		
		V[0].x, V[0].y, V[0].z,
		V[1].x, V[1].y, V[1].z,
		V[2].x, V[2].y, V[2].z);
	
	std::cout << "U Matrix \n";
	print3x3matrix(U);

	std::cout << "V Matrix \n";
	print3x3matrix(V);

	// Compute Rotation U.VT
	matmulUVCPU(3, U, V, Rot);
	std::cout << "Rotation Matrix \n";
	print3x3matrix(Rot);

	// Compute Translation T = Y_mean - R*Xmean
	computeTransCPU(3, mean_x, mean_y, Rot, Trans);

	std::cout << "Translation \n";
	std::cout << Trans.x << " " << Trans.y << " " << Trans.z << " " << std::endl;

	//Compute new X points based on Rot and trans
	updateXCPU(Rot, Trans, Xbuffer, X);

	std::cout << "Updated X \n";
	std::cout << Xbuffer[0].x << " " << Xbuffer[0].y << " " << Xbuffer[0].z << " " << std::endl;

	//Render 
	cudaMemcpy(&dev_pos[Y], &Xbuffer[0], Xbuffer.size()*sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	cudaThreadSynchronize();
	//// Performace Measurment
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float mills = 0.0;
	//cudaEventElapsedTime(&mills, start, stop);
	//// Running Average
	//milliseconds = (milliseccnts*milliseconds + mills) / (milliseccnts + 1);
	//printf("Running Avg Time in milisecond Naive = %f \n", milliseconds);
}


// //===================================
// //==============GPU==================
// //===================================


void Points::initGPU(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer) {
	
	int Y = Ybuffer.size();
	int X = Xbuffer.size();

	numObjects = Y + X;

	// Bring data to GPU
	std::cout << "Setting up gpu buffers ";
	cudaMalloc((void**)&dev_Ybuffer, Y * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_Y failed!");

	cudaMalloc((void**)&dev_Xbuffer, X * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_X failed!");
	std::cout << "...";

	cudaMemcpy(dev_Ybuffer, &Ybuffer[0], Ybuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Xbuffer, &Xbuffer[0], Xbuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcopy Xbuffer Ybuffer failed!");

	cudaMalloc((void**)&dev_YbufferCorr, X*sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_YbufferCorr failed!");

	cudaMalloc((void**)& dev_intermMats, X*sizeof(glm::mat3));
	checkCUDAErrorWithLine("cudaMalloc dev_intermMats failed!");

	cudaMalloc((void**)&dev_Ybuffer_m, X * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_Y failed!");

	cudaMalloc((void**)&dev_Xbuffer_m, X * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_X failed!");

	std::cout <<"Finished." << std::endl;
	cudaThreadSynchronize();
}


__global__ void kernFindCorrespondences(int X, int Y, glm::vec3 *Xbuffer, glm::vec3 *Ybuffer, glm::vec3 *Ybuffer_corr) {

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx < X) {
		
		Ybuffer_corr[idx] = glm::vec3(0.0f, 0.0f, 0.0f);
		float dist = glm::distance(Ybuffer[0], Xbuffer[idx]);

		for (int y = 1; y < Y; y++) {

			float d = glm::distance(Ybuffer[y], Xbuffer[idx]);

			if (dist > d) {
				dist = d;
				Ybuffer_corr[idx] = Ybuffer[y];
			}
		
		}
		//if(idx==0 || idx==X-1)printf("%d CCorr %f %f %f", idx, Ybuffer_corr[idx].x, Ybuffer_corr[idx].y, Ybuffer_corr[idx].z);
	}
}


// Matrix Transpose
__global__ void kernMatrixTranspose(int rows, int cols, float *matrix, float *matrix_T) {

	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < cols && idy < rows) {
		int pos = idy * cols + idx;
		int tpos = idx * rows + idy;

		matrix_T[tpos] = matrix[pos];
	}
}


// kernel to to matmul // A mxn // B nxk // C mxk
__global__ void kernMatrixMultiply(const float *dev_A, const float *dev_B, float *dev_C, int m, int n, int k) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0;
	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
			sum += dev_A[row * n + i] * dev_B[i * k + col];
		dev_C[row * k + col] = sum;
	}
}

__global__ void kernOuterProd(int X, glm::vec3 *dev_Xbuffer, glm::vec3 *dev_YbufferCorr, glm::mat3 *dev_intermMats) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= X) return;
	
	dev_intermMats[idx] = glm::outerProduct( dev_YbufferCorr[idx], dev_Xbuffer[idx]);
}


__global__ void kernMeanCenter(int X, glm::vec3 *dev_buffer, glm::vec3 *dev_buffer_m, glm::vec3 mean) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= X) return;

	dev_buffer_m[idx] = dev_buffer[idx] - mean;
}

__global__ void kernRotTransPoints(int X, glm::vec3 * dev_Xbuffer, glm::mat3 Rot, glm::vec3 Trans) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= X) return;

	dev_Xbuffer[idx] = Trans + glm::vec3(glm::dot(glm::vec3(Rot[0][0], Rot[1][0], Rot[2][0]), dev_Xbuffer[idx]),
									glm::dot(glm::vec3(Rot[0][1], Rot[1][1], Rot[2][1]), dev_Xbuffer[idx]),
									glm::dot(glm::vec3(Rot[0][2], Rot[1][2], Rot[2][2]), dev_Xbuffer[idx]));

}

void Points::stepSimulationGPUNaive(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer, float dt) {

	int Y = Ybuffer.size();
	int X = Xbuffer.size();

	dim3 fullBlocksPerGrid((X + blockSize - 1) / blockSize);


	// Performace Measurment
	// cudaEventCreate(&start);

	//Find Correspondences
	//std::cout << "Computing Correspondances.\n";
	kernFindCorrespondences << <fullBlocksPerGrid, blockSize >> > (X, Y, dev_Xbuffer, dev_Ybuffer, dev_YbufferCorr);
	cudaThreadSynchronize();

	//compute Mean
	glm::vec3 X_mean = thrust::reduce(thrust::device, dev_Xbuffer, dev_Xbuffer + X, glm::vec3(0.0f));
	glm::vec3 Y_mean = thrust::reduce(thrust::device, dev_YbufferCorr, dev_YbufferCorr + X, glm::vec3(0.0f));
	
	X_mean = X_mean / (1.0f * X);
	Y_mean = Y_mean / (1.0f * X);

	//std::cout << " X mean :" << X_mean.x << " " << X_mean.y << " " << X_mean.z << std::endl;
	//std::cout << " Y mean :" << Y_mean.x << " " << Y_mean.y << " " << Y_mean.z << std::endl;

	// MeanCenter Data
	kernMeanCenter << < fullBlocksPerGrid, blockSize >> > (X, dev_Xbuffer, dev_Xbuffer_m, X_mean);
	kernMeanCenter << < fullBlocksPerGrid, blockSize >> > (X, dev_YbufferCorr, dev_Ybuffer_m, Y_mean);
	cudaThreadSynchronize();

	// Compute YtX
	//std::cout << "Computing kernOuterProd \n";

	// ELement wise outer product
	kernOuterProd << <fullBlocksPerGrid, blockSize >> > (X, dev_Xbuffer_m, dev_Ybuffer_m, dev_intermMats);
	//std::cout << "Computing thrust::reduce \n";
	cudaThreadSynchronize();

	//Reduction using Thrust
	glm::mat3 YtX = thrust::reduce(thrust::device, dev_intermMats, dev_intermMats + X, glm::mat3(0.0f));

	
	//std::cout << "Matrix YtX\n";
	//std::cout << YtX[0][0] << " " << YtX[1][0] << " " << YtX[2][0] << "\n"
			 // << YtX[0][1] << " " << YtX[1][1] << " " << YtX[2][1] << "\n"
			 // << YtX[0][2] << " " << YtX[1][2] << " " << YtX[2][2] << "\n";
	

	//cudaMemcpy(&dev_matrix1, &YtX, sizeof(glm::mat3), cudaMemcpyHostToDevice);

	// ComputeMpute SVD(YtX)
	glm::mat3 Rot(0.0f);
	glm::vec3 Trans(0.0f);

	glm::mat3 U(0.0f);
	glm::mat3 S(0.0f);
	glm::mat3 V(0.0f);

	// compute SVD

	svd(YtX[0][0], YtX[1][0], YtX[2][0],
		YtX[0][1], YtX[1][1], YtX[2][1],
		YtX[0][2], YtX[1][2], YtX[2][2],

		U[0][0], U[1][0], U[2][0],
		U[0][1], U[1][1], U[2][1],
		U[0][2], U[1][2], U[2][2],

		S[0][0], S[1][0], S[2][0],
		S[0][1], S[1][1], S[2][1],
		S[0][2], S[1][2], S[2][2],

		V[0][0], V[1][0], V[2][0],
		V[0][1], V[1][1], V[2][1],
		V[0][2], V[1][2], V[2][2] );

	//Compute R = UVt
	Rot = U * glm::transpose(V);
	/*
	std::cout << "Matrix Rotation \n";
	std::cout << Rot[0][0] << " " << Rot[1][0] << " " << Rot[2][0] << "\n"
			  << Rot[0][1] << " " << Rot[1][1] << " " << Rot[2][1] << "\n"
			  << Rot[0][2] << " " << Rot[1][2] << " " << Rot[2][2] << "\n";
	*/
	//compute T = Ymean =RXmean
	Trans = Y_mean - (Rot * X_mean) ;
	/*
	std::cout << "Trans \n";
	std::cout << Trans.x << " " << Trans.y << " " << Trans.z << " " << std::endl;
	*/
	// Update dev_Xbuffer
	kernRotTransPoints <<< fullBlocksPerGrid, blockSize >> > (X, dev_Xbuffer, Rot, Trans); //RX + T
	cudaThreadSynchronize();


	glm::vec3 Xbuff(0.0f);
	cudaMemcpy(&Xbuff, &dev_Xbuffer[0], sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcopy Xbuff failed!");
	
	/*
	std::cout << "Updated X \n";
	std::cout << Xbuff.x << " " << Xbuff.y << " " << Xbuff.z << " " << std::endl;
	*/


	//render
	cudaMemcpy(&dev_pos[Y], &dev_Xbuffer[0], X * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcopy Xbuff failed!");
	cudaThreadSynchronize();

	//// Performace Measurment
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float mills = 0.0;
	cudaEventElapsedTime(&mills, start, stop);
	printf("Points::stepSimulationGPUNaive | Time-milisecond = %f \n", mills);
*/
}

void Points::endSimulation() {
	cudaFree(dev_col);
	cudaFree(dev_Ybuffer);
	cudaFree(dev_Xbuffer);
	cudaFree(dev_Xbuffer_m);
	cudaFree(dev_Ybuffer_m);
	cudaFree(dev_YbufferCorr);
	cudaFree(dev_pos);
	cudaFree(dev_intermMats);

	checkCUDAErrorWithLine("cudaFree failed!");
}

void Points::unitTest() {
	// LOOK-1.2 Feel free to write additional tests here.
	// test unstable sort
	int *dev_intKeys;
	int *dev_intValues;
	int N = 10;

	std::unique_ptr<int[]>intKeys{ new int[N] };
	std::unique_ptr<int[]>intValues{ new int[N] };

	intKeys[0] = 0; intValues[0] = 0;
	intKeys[1] = 1; intValues[1] = 1;
	intKeys[2] = 0; intValues[2] = 2;
	intKeys[3] = 3; intValues[3] = 3;
	intKeys[4] = 0; intValues[4] = 4;
	intKeys[5] = 2; intValues[5] = 5;
	intKeys[6] = 2; intValues[6] = 6;
	intKeys[7] = 0; intValues[7] = 7;
	intKeys[8] = 5; intValues[8] = 8;
	intKeys[9] = 6; intValues[9] = 9;

	cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

	cudaMalloc((void**)&dev_intValues, N * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

	std::cout << "before unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// How to copy data to the GPU
	cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

	// Wrap device vectors in thrust iterators for use with thrust.
	thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
	thrust::device_ptr<int> dev_thrust_values(dev_intValues);

	// LOOK-2.1 Example for using thrust::sort_by_key
	thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

	// How to copy data back to the CPU side from the GPU
	cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("memcpy back failed!");

	std::cout << "after unstable sort: " << std::endl;
	for (int i = 0; i < N; i++) {
		std::cout << "  key: " << intKeys[i];
		std::cout << " value: " << intValues[i] << std::endl;
	}

	// cleanup
	cudaFree(dev_intKeys);
	cudaFree(dev_intValues);
	checkCUDAErrorWithLine("cudaFree failed!");
	return;
}
