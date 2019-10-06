#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace Points {
    void initSimulation(std::vector<glm::vec3> &Ybuffer, std::vector<glm::vec3>&Xbuffer);
	void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities);


	void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);

    void endSimulation();
    void unitTest();
}
