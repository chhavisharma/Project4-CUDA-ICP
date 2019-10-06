#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
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
#define blockSize 128

// LOOK-1.2 Parameters for the points algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 0.10f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your point information.
// These get allocated for you in Points::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// point cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?

// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_pos_shuffle;
glm::vec3 *dev_vel1_shuffle;

cudaEvent_t start, stop;

float milliseconds = 0;
int milliseccnts = 0;
// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Points::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
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

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating points with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals

void Points::initSimulation(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer) {
  
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Points::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");
  
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dev_thrust_particleGridIndices = thrust::device_pointer_cast<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);

  cudaMalloc((void**)&dev_pos_shuffle, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_shuffle failed!");

  cudaMalloc((void**)&dev_vel1_shuffle, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_shuffle failed!");

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
}

*/

__global__ void kernResetVec3Buffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


void Points::initSimulation(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer) {

	int Y = Ybuffer.size();
	int X = Xbuffer.size();

	numObjects = Y + X;

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_vel1, numObjects * sizeof(glm::vec3));
	checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

	cudaMemcpy(dev_pos, &Ybuffer[0], Ybuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_pos[Ybuffer.size()], &Xbuffer[0], Xbuffer.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	kernResetVec3Buffer <<<dim3((Y + blockSize - 1) / blockSize), blockSize >> > (Y, dev_vel1, glm::vec3(1, 1, 1));
	kernResetVec3Buffer <<<dim3((X + blockSize - 1) / blockSize), blockSize >> > (X, &dev_vel1[Y], glm::vec3(0, 1, 0));

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

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyPointsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` points
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	
	glm::vec3 pCenter(0.0f, 0.0f, 0.0f);
	glm::vec3 pDist(0.0f, 0.0f, 0.0f);
	glm::vec3 pVel(0.0f, 0.0f, 0.0f);
	int pcounter = 0;
	int vcounter = 0;

	for (int b = 0; b < N; b++) {
		
		// Rule 1: points fly towards their local perceived center of mass
		if (b != iSelf && glm::distance(pos[b],pos[iSelf]) < rule1Distance) {
			pCenter += pos[b];
			pcounter += 1;
		}
		// Rule 2: points try to stay a distance d away from each other
		if( b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule2Distance) {
			pDist -= (pos[b] - pos[iSelf]);
		}
		// Rule 3: points try to match the speed of surrounding points
		if(b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule3Distance) {
			pVel += vel[b];
			vcounter += 1;
		}
	}
	
	// Accumuate values
	if (pcounter != 0) {
		pCenter /= pcounter;
		pCenter = (pCenter - pos[iSelf])*rule1Scale;
	}

	pDist = pDist * rule2Scale;

	if (vcounter != 0) {
		pVel /= vcounter;
		pVel = pVel * rule3Scale;
	}
	
	// Return overall change in velocity!
	return pCenter + pDist + pVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
		
	glm::vec3 cvel(0.0f, 0.0f, 0.0f);
	
	//Reslove Boid's index
	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	// compute new velocity using all 3 rules!
	if (iSelf < N) {
		cvel = computeVelocityChange(N, iSelf, pos, vel1);
		//printf("cval %f %f %f", (cvel.x, cvel.y, cvel.z));
		// Clamp max speed
		// cvel = glm::clamp(cvel,-maxSpeed, maxSpeed);
		// Record the new velocity into vel2. Question: why NOT vel1?
		// Becaase we want all threads to finish computing new vel before updates are made.
		// This new ve lcomputation is dependent on other point-velocitites		
		
		vel2[iSelf] = cvel + vel1[iSelf];

		vel2[iSelf] = glm::clamp(vel2[iSelf], -maxSpeed, maxSpeed);
	}
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the points around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1

	// - Label each point with the index of its grid cell.
	//Reslove Boid's index
	// - Set up a parallel array of integer indices as pointers to the actual
	//   point data in pos and vel1/vel2

	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iSelf < N) {
		glm::vec3 grid_ind  = glm::floor((pos[iSelf] - gridMin) * inverseCellWidth);
		gridIndices[iSelf]  = (int)gridIndex3Dto1D((int)grid_ind.x, (int)grid_ind.y, (int)grid_ind.z, gridResolution);
		indices[iSelf]		= iSelf;
	}
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any points
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iSelf < N) {
		if (iSelf == 0) {
			gridCellStartIndices[particleGridIndices[0]] = 0;
		}
		else if (iSelf == N-1) {
			gridCellEndIndices[particleGridIndices[N-1]] = N-1;
		}
		else if(particleGridIndices[iSelf]!= particleGridIndices[iSelf-1]){

			gridCellEndIndices[particleGridIndices[iSelf - 1]]= iSelf - 1;
			gridCellStartIndices[particleGridIndices[iSelf]] = iSelf;
		}
	}
}

__device__ int gridsToCheck(int x, int y, int z, int gridResolution) {
	return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
	// TODO-2.1 - Update a point's velocity using the uniform grid to reduce
	// the number of points that need to be checked.

	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (iSelf < N) {
		
		// - Identify the grid cell that this particle is in
		glm::vec3 gird3Didx = glm::floor((pos[iSelf]-gridMin) * inverseCellWidth);
		//printf("gird3Didx %f %f %f \n", gird3Didx.x, gird3Didx.y, gird3Didx.z);

		// Variables to hold accumulated vel per rule
		glm::vec3 pCenter(0.0f, 0.0f, 0.0f);
		glm::vec3 pVel(0.0f, 0.0f, 0.0f);
		glm::vec3 pDist(0.0f, 0.0f, 0.0f);

		int cnt_pCenter = 0;
		int cnt_pVel = 0;

		//int x_min = (int)gird3Didx.x - 1, int y_min = (int)gird3Didx.y - 1; int z_min = (int)gird3Didx.z -1;
		//int x_max = (int)gird3Didx.x + 1, int y_max = (int)gird3Didx.y + 1; int z_max = (int)gird3Didx.z +1;

		// variable sphere limit to get candidate cells
		float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);

		glm::vec3 gird3D_max = glm::floor((pos[iSelf] - gridMin + maxDistance) * inverseCellWidth);
		glm::vec3 gird3D_min = glm::floor((pos[iSelf] - gridMin - maxDistance) * inverseCellWidth);

		int x_max = gird3D_max.x; int y_max = (int)gird3D_max.y, int z_max = (int)gird3D_max.z;
		int x_min = gird3D_min.x; int y_min = (int)gird3D_min.y, int z_min = (int)gird3D_min.z;

		for (int x = x_min; x <= x_max; x++) {
			for (int y = y_min; y <= y_max; y++) {
				for (int z = z_min; z <= z_max; z++) {
					if (x >= 0 && y >= 0 && z >= 0 && x < gridResolution && y < gridResolution && z < gridResolution) {


						// - For each cell, read the start/end indices in the point pointer array.
						int grid1Didx = gridIndex3Dto1D(x, y, z, gridResolution);
						int strt_idx = gridCellStartIndices[grid1Didx];
						int end_idx = gridCellEndIndices[grid1Didx];

						if (strt_idx < 0 || end_idx < 0 || strt_idx >= N || end_idx >= N) continue;

						for (int grid_idx = strt_idx; grid_idx <= end_idx; grid_idx++) {
							int b = particleArrayIndices[grid_idx];
								
							// Rule 1: points fly towards their local perceived center of mass, which excludes themselves
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule1Distance) {
								pCenter += pos[b];
								cnt_pCenter += 1;
							}
							// Rule 2: points try to stay a distance d away from each other
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule2Distance) {
								pDist -= (pos[b] - pos[iSelf]);
							}
							// Rule 3: points try to match the speed of surrounding points
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule3Distance) {
								pVel += vel1[b];
								cnt_pVel += 1;
							}
						}
					}
				}
			}
		}

		// Accumulated Avg
		if (cnt_pCenter != 0) {
			pCenter /= cnt_pCenter;
			pCenter = (pCenter - pos[iSelf])*rule1Scale;
		}
		
		pDist = pDist * rule2Scale;

		if (cnt_pVel != 0) {
			pVel /= cnt_pVel;
			pVel = pVel * rule3Scale;
		}

		// - Clamp the speed change before putting the new speed in vel2
		vel2[iSelf] = pCenter + pDist + pVel + vel1[iSelf];
		vel2[iSelf] = glm::clamp(vel2[iSelf], -maxSpeed, maxSpeed);
	}
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the point pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the points data.
  // - Access each point in the cell and compute velocity change from
  //   the points rules, if this point is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iSelf < N) {
		glm::vec3 gird3Didx = glm::floor((pos[iSelf] - gridMin) * inverseCellWidth);

		glm::vec3 pCenter(0.0f, 0.0f, 0.0f);
		glm::vec3 pVel(0.0f, 0.0f, 0.0f);
		glm::vec3 pDist(0.0f, 0.0f, 0.0f);

		int cnt_pCenter = 0;
		int cnt_pVel = 0;

		//int x_min = (int)gird3Didx.x - 1, int y_min = (int)gird3Didx.y - 1, int z_min = (int)gird3Didx.z -1;
		//int x_max = (int)gird3Didx.x + 1, int y_max = (int)gird3Didx.y + 1, int z_max = (int)gird3Didx.z +1;

		// variable sphere limit to get candidate cells
		float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);

		glm::vec3 gird3D_max = glm::floor((pos[iSelf] - gridMin + maxDistance) * inverseCellWidth);
		glm::vec3 gird3D_min = glm::floor((pos[iSelf] - gridMin - maxDistance) * inverseCellWidth);

		int x_max = gird3D_max.x; int y_max = (int)gird3D_max.y; int z_max = (int)gird3D_max.z;
		int x_min = gird3D_min.x; int y_min = (int)gird3D_min.y; int z_min = (int)gird3D_min.z;
		//int CellCheckCounter = 0;
		
		for (int x = x_min; x <= x_max; x++) {
			for (int y = y_min; y <= y_max; y++) {
				for (int z = z_min; z <= z_max; z++) {
					if (x >= 0 && y >= 0 && z >= 0 && x < gridResolution && y < gridResolution && z < gridResolution) {
						//CellCheckCounter += 1;

						// - For each cell, read the start/end indices in the point pointer array.
						int grid1Didx = gridIndex3Dto1D(x, y, z, gridResolution);
						int strt_idx = gridCellStartIndices[grid1Didx];
						int end_idx = gridCellEndIndices[grid1Didx];

						if (strt_idx < 0 || end_idx <0 || strt_idx >= N|| end_idx >= N ) continue;

						for (int b = strt_idx; b <= end_idx; b++) {

							// Rule 1: points fly towards their local perceived center of mass, which excludes themselves
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule1Distance) {
								pCenter += pos[b];
								cnt_pCenter += 1;
							}

							// Rule 2: points try to stay a distance d away from each other
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule2Distance) {
								pDist -= (pos[b] - pos[iSelf]);
							}

							// Rule 3: points try to match the speed of surrounding points
							if (b != iSelf && glm::distance(pos[b], pos[iSelf]) < rule3Distance) {
								pVel += vel1[b];
								cnt_pVel += 1;
							}
						}
					}
				}
			}
		}

		//printf("iself %d | Cell check counter = %d \n",(iSelf, CellCheckCounter));

		// Accumulated Avg
		if (cnt_pCenter != 0) {
			pCenter /= cnt_pCenter;
			pCenter = (pCenter - pos[iSelf])*rule1Scale;
		}

		pDist = pDist * rule2Scale;

		if (cnt_pVel != 0) {
			pVel /= cnt_pVel;
			pVel = pVel * rule3Scale;
		}

		// - Clamp the speed change before putting the new speed in vel2
		vel2[iSelf] = pCenter + pDist + pVel + vel1[iSelf];
		vel2[iSelf] = glm::clamp(vel2[iSelf], -maxSpeed, maxSpeed);
	}
}

__global__ void kernShuffleVelPosCoherent(
	int N, int *particleArrayIndices,
	glm::vec3 *pos, glm::vec3 *vel1,
	glm::vec3 *pos_shuffle, glm::vec3 *vel1_shuffle) {
	
	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iSelf < N) {
		pos_shuffle[iSelf] = pos[particleArrayIndices[iSelf]];
		vel1_shuffle[iSelf] = vel1[particleArrayIndices[iSelf]];
		}
}

__global__ void kernUnShuffleVelPosCoherent(
	int N, int *particleArrayIndices,
	glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2_shuffle) {
	
	int iSelf = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iSelf < N) {
		vel1[particleArrayIndices[iSelf]] = vel2_shuffle[iSelf];
	}
}


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Points::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// Sequentially call both kernels

	// Performace Measurment 
	//cudaEventRecord(start);

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	kernUpdateVelocityBruteForce <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

	kernUpdatePos<<< fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// TODO-1.2 ping-pong the velocity buffers
	dev_vel1 = dev_vel2;

	//// Performace Measurment 
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);

	//float mills = 0.0;
	//cudaEventElapsedTime(&mills, start, stop);

	//// Running Average
	//milliseconds = (milliseccnts*milliseconds + mills) / (milliseccnts + 1);
	//printf("Running Avg Time in milisecond Naive = %f \n", milliseconds);

}

void Points::stepSimulationScatteredGrid(float dt) {
	// TODO-2.1
	// Uniform Grid Neighbor search using Thrust sort.
	// In Parallel:
	// - label each particle with its array index as well as its grid index.
	// - Unstable key sort using Thrust. 
	// Wrap device vectors in thrust iterators for use with thrust.
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of point indices

	// Performace Measurment 
	// cudaEventRecord(start);

	int N = numObjects;

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);
	
	kernResetIntBuffer << < cellBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellStartIndices failed!");

	kernResetIntBuffer << < cellBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellEndIndices failed!");

	kernComputeIndices <<< fullBlocksPerGrid, blockSize >>> (N, gridSideCount, gridMinimum, gridInverseCellWidth, 
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);
	
	kernIdentifyCellStartEnd <<< fullBlocksPerGrid, blockSize >> > (N, dev_particleGridIndices, 
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	kernUpdateVelNeighborSearchScattered <<< fullBlocksPerGrid, blockSize >> > (N, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices, 
		dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (N, dt, dev_pos, dev_vel2);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	dev_vel1 = dev_vel2;

	// Performace Measurment 
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);

	//float mills = 0.0;
	//cudaEventElapsedTime(&mills, start, stop);

	// Running Average
	//milliseconds = (milliseccnts*milliseconds + mills) / (milliseccnts + 1);
	//if(milliseccnts%10000==0) printf("Time in milisecond Scattered = %f \n", mills);
}

void Points::stepSimulationCoherentGrid(float dt) {
	// TODO-2.3 - start by copying Points::stepSimulationNaiveGrid
  
	// Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  
	// In Parallel:
	// - Label each particle with its array index as well as its grid index.
	//   Use 2x width grids
	// - Unstable key sort using Thrust. A stable sort isn't necessary, but you
	//   are welcome to do a performance comparison.
	// - Naively unroll the loop for finding the start and end indices of each
	//   cell's data pointers in the array of point indices
	// - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
	//   the particle data in the simulation array.
	//   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
	// - Perform velocity updates using neighbor search
	// - Update positions
	// - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
	
	int N = numObjects;

	dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);
	dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);

	// Performace Measurment 
	//cudaEventRecord(start);
	//printf("fullBlocksPerGrid %d \n", ((N + blockSize - 1) / blockSize));
	//printf("blockSize		  %d \n", blockSize);


	kernResetIntBuffer << < cellBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellStartIndices failed!");

	kernResetIntBuffer << < cellBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer for dev_gridCellEndIndices failed!");

	kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (N, gridSideCount, gridMinimum, gridInverseCellWidth,
		dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + N, dev_thrust_particleArrayIndices);

	kernShuffleVelPosCoherent <<< fullBlocksPerGrid, blockSize >>> (N, dev_particleArrayIndices,
			dev_pos, dev_vel1, dev_pos_shuffle, dev_vel1_shuffle);

	kernIdentifyCellStartEnd << < fullBlocksPerGrid, blockSize >> > (N, dev_particleGridIndices,
		dev_gridCellStartIndices, dev_gridCellEndIndices);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	kernUpdateVelNeighborSearchCoherent <<< fullBlocksPerGrid, blockSize >> > (N, gridSideCount, gridMinimum,
		gridInverseCellWidth, gridCellWidth,
		dev_gridCellStartIndices, dev_gridCellEndIndices,
		dev_pos_shuffle, dev_vel1_shuffle, dev_vel2);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	kernUnShuffleVelPosCoherent<<<fullBlocksPerGrid, blockSize >>> (N, dev_particleArrayIndices,
		dev_pos, dev_vel1, dev_vel2);

	kernUpdatePos << < fullBlocksPerGrid, blockSize >> > (N, dt, dev_pos, dev_vel1);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	//// Performace Measurment 
	//cudaEventRecord(stop);

	//cudaEventSynchronize(stop);

	//float mills= 0.0;
	//cudaEventElapsedTime(&mills, start, stop);
	//
	//// Running Average
	//milliseconds = (milliseccnts*milliseconds + mills)/( milliseccnts + 1);
	//printf("Running Avg Time in milisecond Coherent = %f \n", milliseconds);
}

void Points::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  // cleanup
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos_shuffle);
  cudaFree(dev_vel1_shuffle);
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
