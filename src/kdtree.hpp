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
#include "common.h"

namespace KDTree {
	void initCpuKDTree(std::vector<glm::vec3> &ybuff, std::vector<glm::vec4>& ybufftree);
	unsigned int nextPowerOf2(unsigned int n);
}