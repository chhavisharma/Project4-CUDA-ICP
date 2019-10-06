#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "picp.h"
#include <vector>
#include <algorithm>

namespace ParallelICP {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }


	//==CREATE POINTERS==
	float * Ybuffer_corr;

	float distance3D(float x1, float y1, float z1, float x2, float y2, float z2) {
		
		float dist = (x1 - x2)*(x1 - x2) + (x1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
	
		return dist;
	}


	void findCorrespondences(int Y, int X, float * Ybuffer_corr, float * Ybuffer, float * Xbuffer) {
		
		for (int i = 0; i < X; i++) { // for each X
			
			Ybuffer_corr[i] = 0;
			float dist = distance3D(Ybuffer[dims*0 + 0], Ybuffer[dims*0 + 1], Ybuffer[dims*0 + 2],
									Xbuffer[dims*i + 0], Xbuffer[dims*i + 1], Xbuffer[dims*i + 2]);

			for (int j = 1; j < Y; j++) { // all Y's

				float d = distance3D(Ybuffer[dims*j + 0], Ybuffer[dims*j + 1], Ybuffer[dims*j + 2],
									 Xbuffer[dims*i + 0], Xbuffer[dims*i + 1], Xbuffer[dims*i + 2]);
				if (dist > d ){
					dist = d;
					Ybuffer_corr[i] = j;
					}
			}
		}
	
	}


	void computeICPNaive(int Y, int X, float * Ybuffer, float * Xbuffer) {
		
		//=====================================
		//=======  FIND CORRESPONDENCES =======
		//=====================================
		Ybuffer_corr = new float[X * dims * sizeof(float)];
		findCorrespondences(Y, X, Ybuffer_corr, Ybuffer, Xbuffer);

		
		//=====================================
		//==============  RANSAC  =============
		//=====================================	



	;
	}
}
