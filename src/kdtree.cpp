#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <algorithm>

#include "kdtree.hpp"

#define X 0
#define Y 1
#define Z 2

struct sort1{
	bool operator() (const glm::vec3 &i, const glm::vec3 &j) { return (i.x < j.x); }
} sortbyX;

struct sort2{
	bool operator() (const glm::vec3 &i, const glm::vec3 &j) { return (i.y < j.y); }
} sortbyY;

struct sort3 {
	bool operator() (const glm::vec3 &i, const glm::vec3 &j) { return (i.z < j.z); }
} sortbyZ;


bool sortbyx(const glm::vec3 &i, const glm::vec3 &j) { return (i.x < j.x); }
bool sortbyy(const glm::vec3 &i, const glm::vec3 &j) { return (i.y < j.y); }
bool sortbyz(const glm::vec3 &i, const glm::vec3 &j) { return (i.z < j.z); }


void printKDTree3(std::vector<glm::vec3> &tree) {
	for (int i = 0; i < tree.size(); i++) {
		std::cout << tree[i].x << " " << tree[i].y << " "<< tree[i].z << "\n";
	}
	std::cout << "\n";
}

void printKDTree4(std::vector<glm::vec4> &tree) {
	for (int i = 0; i < tree.size(); i++) {
		std::cout << tree[i].x << " " << tree[i].y << " " << tree[i].z<< " "<< tree[i].w << "\n";
	}
	std::cout << "\n";
}

unsigned int KDTree::nextPowerOf2(unsigned int n)
{
	unsigned count = 0;
	if (n && !(n & (n - 1)))
		return n;

	while (n != 0)
	{
		n >>= 1;
		count += 1;
	}
	return 1 << count;
}

void buildTree(std::vector<glm::vec3> &ybuff, int start, int end, int pidx, std::vector<glm::vec4>& ybufftree, int depth) {
	
	if(start <= end) {
		//std::cout << "depth=" << depth;

		// Get current depth dim
		int d = depth % dimension;

		//std::cout << " dims=" << d << "\n";

		// Sort by dim
		if (d == X) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyX); }
		else if (d == Y) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyY); }
		else if (d == Z) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyZ); }

		//std::cout << "Finished sorting !\n";

		// pick median 
		int mid = (start + end) / 2;

		//std::cout << "Mid is = " << mid << "\n";

		// store the parent // set not null in w 
		ybufftree[pidx].x = ybuff[mid].x;
		ybufftree[pidx].y = ybuff[mid].y;
		ybufftree[pidx].z = ybuff[mid].z;
		ybufftree[pidx].w = 1.0f;

		//std::cout << "Set at pidx = " << pidx << " | vales =" << ybuff[mid].x << " " << ybuff[mid].y << " " << ybuff[mid].z << "\n";

		//build tree for left and right child

		// Left Child
		buildTree(ybuff, start, mid - 1, (2 * pidx + 1), ybufftree, depth + 1);

		// Right Child
		buildTree(ybuff, mid + 1, end, (2 * pidx + 2), ybufftree, depth + 1);
	}
	return;
}


void mystack_pushCPU(std::vector<mystack>& st, int data, bool good, int depth, int &top) {
	top += 1;
	st[top].dataIdx = data;
	st[top].good = good;
	st[top].depth = depth;
}

void mystack_popCPU(std::vector<mystack>& st, int &data, bool &good, int &depth, int &top) {
	data = st[top].dataIdx;
	good = st[top].good ;
	depth = st[top].depth;
	top -= 1;
}

float compute_distanceCPU(glm::vec3 query, glm::vec3 target, int depth, bool &right) {

	int d = depth% dimension;

	float dist = 0.0f;
	if (d == X) {
		dist = std::abs(query.x - target.x);
		right = query.x > target.x ? true : false;
	}
	else if (d == Y) { 
		dist = std::abs(query.y - target.y);
		right = query.y > target.y ? true : false;
	}
	else if (d == Z) { 
		dist = std::abs(query.z - target.z);
		right = query.z > target.z ? true : false;
	}
	return dist;
}

void KDclosestPointCPU(std::vector<glm::vec4>& ybufftree, glm::vec3 goal, int &idx, float &dist, std::vector<mystack> &st) {

	int didx   = 0;
	int depth = 0;
	int top   = 0;
	bool kind = false;
	
	bool good_root = false;

	int bestIdx = 0;
	float bestDist = 1.0f*LONG_MAX;
	
	mystack_pushCPU(st, didx, true, depth, top);
	std::cout << "Post Pre Push Root | didx=" << didx << " top=" << top << " depth=" << depth << std::endl;

	// simluating recursion 
	while (top > 0) {
		
		//didx, depth, kind, top all by reference
		mystack_popCPU(st, didx, good_root, depth, top);
		
		std::cout << "==============================================================\n";
		std::cout << "Init Pop | didx=" << didx << " top=" << top << " good="<<good_root<<" depth=" << depth << " isNotNull=" << ybufftree[didx].w << std::endl;
		
		if (ybufftree[didx].w == 0.0f){ // You are null - Exit
			std::cout << "I am Null " << std::endl;
			continue;
		}

		//get current point data
		glm::vec3 currPoint(ybufftree[didx].x, ybufftree[didx].y, ybufftree[didx].z);
		std::cout << "Got current point" << std::endl;
		
		// compute distance and goleft / goRight
		bool right = false;
		float dist = compute_distanceCPU(goal, currPoint, depth, right);
		std::cout << "Computed Distance" << std::endl;
				
		if (good_root == false && bestDist < dist ) {
			std::cout << "Bad Node and bestDist<dist " << bestDist <<"<"<<dist << " So COntinue!"<< std::endl;
			continue; 
		}
		
		// update best distance 
		if (bestDist > dist) { 
			bestDist = dist;
			bestIdx = didx;
		}
		
		// Now look at your children!

		int depth_children = depth + 1;

		if (right == true) { // goodSide is Right, // Badside is left
			//bad child == left
			mystack_pushCPU(st, 2* didx +1, false, depth_children, top);
			std::cout << "Pushed bad child at didx" << 2 * didx + 1 << " depth_children " << depth_children << " top becomes=" << top<< std::endl;
			//good child == right
			mystack_pushCPU(st, 2* didx +2, true, depth_children, top);
			std::cout << "Pushed good child at didx" << 2 * didx + 2 << " depth_children " << depth_children << " top becomes=" << top << std::endl;
		}
		else {
			//bad child == right
			mystack_pushCPU(st, 2*didx+2, false, depth_children, top);
			std::cout << "Pushed bad child at didx" << 2 * didx + 2 << " depth_children " << depth_children << " top becomes=" << top << std::endl;

			//good child == left
			mystack_pushCPU(st, 2*didx+1, true, depth_children, top);
			std::cout << "Pushed good child at didx" << 2 * didx + 1 << " depth_children " << depth_children << " top becomes=" << top << std::endl;
		}
	}
	std::cout << "\n\nFinished traversing! Top=" <<top <<std::endl;
	idx = bestIdx;
	dist = bestDist;
}

void KDTree::initCpuKDTree(std::vector<glm::vec3> &ybuff, std::vector<glm::vec4>&ybufftree) {

	// Inits
	int n = ybuff.size();
	int parentIdx = 0;
	int depth = 0;
	
	//std::cout << "Pritnting input \n";
	//printKDTree3(ybuff);

	// Compute the KDTree
	buildTree(ybuff, 0, n-1, parentIdx, ybufftree, depth);
	std::cout << "Computed KD tree \n";
	//std::cout << "Pritnting level order kd tree \n";
	//printKDTree4(ybufftree);

	// Testing KD tree
	//int idx = -1;
	//float dist = -1.0f;

	////glm::vec3 goal(7.1, 1.1, 4.1);
	//glm::vec3 goal(1.1, 7.1, 4.8);
	//std::cout << "Searching the kd tree for point \n";
	//std::cout << goal.x << " " << goal.y << " " << goal.z << std::endl;
	//
	//std::vector<mystack> stk(ybufftree.size());
	//KDclosestPointCPU(ybufftree, goal, idx, dist, stk);
	//
	//std::cout << "Closest Point at idx "<<idx<<" with dist "<< dist << std::endl;
	//std::cout << ybufftree[idx].x<<" " << ybufftree[idx].y<< " " << ybufftree[idx].z << std::endl;

	return;
}