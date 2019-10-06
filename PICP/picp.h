#pragma once

#include "common.h"

namespace ParallelICP {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here

	void computeICPNaive(int Y, int X, float * Ybuffer, float * Xbuffer);

}
