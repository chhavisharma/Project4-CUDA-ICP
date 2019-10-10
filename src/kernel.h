#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include "common.h"

namespace Points {
    void initCpuICP(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer);
	void stepSimulationICPNaive(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer, float dt);
	
	void initGPU(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer);
	void stepSimulationGPUNaive(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer, float dt);

	void initGPUKD(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3> &Xbuffer, glm::vec4*YbufferTree,
		glm::ivec3* track, int treesize, int tracksize);

	void stepSimulationGPUKD(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer,
		 int treesize, int tracksize , float dt);

	void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
    void endSimulation();
    void unitTest();
}
