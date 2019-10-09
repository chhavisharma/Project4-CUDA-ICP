#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#include <iostream>
#include <cmath>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace KDTree {
	//std::vector<glm::vec3> YbufferTree;
	void initCpuKDTree(std::vector<glm::vec3> &ybuff, std::vector<glm::vec4>& ybufftree);
	unsigned int nextPowerOf2(unsigned int n);
	void KDclosestPoint(std::vector<glm::vec4>& ybufftree, glm::vec3 point, int &idx);
}