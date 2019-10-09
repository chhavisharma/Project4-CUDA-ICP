#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <algorithm>

#include "kdtree.hpp"

#define dimension 3
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
		std::cout << "depth=" << depth;

		// Get current depth dim
		int d = depth % dimension;

		std::cout << " dims=" << d << "\n";

		// Sort by dim
		if (d == X) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyX); }
		else if (d == Y) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyY); }
		else if (d == Z) { std::sort(ybuff.begin() + start, ybuff.begin() + end + 1, sortbyZ); }

		std::cout << "Finished sorting !\n";

		// pick median 
		int mid = (start + end) / 2;

		std::cout << "Mid is = " << mid << "\n";

		// store the parent // set not null in w 
		ybufftree[pidx].x = ybuff[mid].x;
		ybufftree[pidx].y = ybuff[mid].y;
		ybufftree[pidx].z = ybuff[mid].z;
		ybufftree[pidx].w = 1.0f;

		std::cout << "Set at pidx = " << pidx << " | vales =" << ybuff[mid].x << " " << ybuff[mid].y << " " << ybuff[mid].z << "\n";

		//build tree for left and right child

		// Left Child
		buildTree(ybuff, start, mid - 1, (2 * pidx + 1), ybufftree, depth + 1);

		// Right Child
		buildTree(ybuff, mid + 1, end, (2 * pidx + 2), ybufftree, depth + 1);
	}
	return;
}
/*
void KDTree::KDclosestPoint(std::vector<glm::vec4>& ybufftree, glm::vec3 goal, int &idx) {

	int start = 0;
	int end = ybufftree.size() - 1;

	int depth = 0;


	int curIdx = 0;
	int nextIdx = 0;

	int bestIdx = 0;
	int bestIdxParent = 0;

	float bestDist = glm::distance(goal, glm::vec3(ybufftree[0].x, ybufftree[0].y, ybufftree[0].z));

	bool finished = false;
	bool nodeDone = false;
	bool goLeft = true; // else goRight
	
	//iterative KDtree search
	while (!finished) {

		// on curIdx
		glm::vec3 curPoint = glm::vec3(ybufftree[curIdx].x, ybufftree[curIdx].y, ybufftree[curIdx].z);
		int dim = dimension % depth;

		//search current node using DFS
		while (ybufftree[curIdx].w != 0.0f) { // currNode is not null

			float dist = glm::distance(goal, curPoint);
			if (dist < bestDist) {
				bestDist = dist;
				bestIdx = curIdx;
				nodeDone = false;
			}

			// pick which child to look at 
			if (dim == X) { goLeft = sortbyx(curPoint, goal); }
			else if (dim == Y) { goLeft = sortbyy(curPoint, goal); }
			else if (dim == Z) { goLeft = sortbyz(curPoint, goal); }

			// find which child to go to
			// nextIdx = goLeft ? leftChild : rightChild
			nextIdx = goLeft ? 2*curIdx + 1 : 2*curIdx + 2;

			curIdx = nextIdx;
		}
		if (nodeDone) {
			finished = true;
		}
		else {
			// check if parent of best node could have better values on other branch
			float d = 0.0f;

			if (dim == X) { d = std::abs(curPoint.x -  goal.x); }
			else if (dim == Y) { d = std::abs(curPoint.y - goal.y); }
			else if (dim == Z) { d = std::abs(curPoint.z - goal.z); }

			if (dim == X) { goLeft = sortbyx(curPoint, goal); }
			else if (dim == Y) { goLeft = sortbyy(curPoint, goal); }
			else if (dim == Z) { goLeft = sortbyz(curPoint, goal); }

			if (d < bestDist) {
				curIdx = goLeft ? 2 * curIdx + 1 : 2 * curIdx + 2;
				nodeDone = true;
			}
			else
				finished = true;
		}

	}
	idx = bestIdx;
}
*/

struct stack {
	int dataIdx;
	bool good;
	int depth;
	stack() : dataIdx(0),good(true),depth(0){}
};

void stack_push(std::vector<stack>& st, int data, bool good, int depth, int &top) {
	top += 1;
	st[top].dataIdx = data;
	st[top].good = good;
	st[top].depth = depth;
}

void stack_pop(std::vector<stack>& st, int &data, bool &good, int &depth, int &top) {
	data = st[top].dataIdx;
	good = st[top].good ;
	depth = st[top].depth;
	top -= 1;
}

float compute_distance(glm::vec3 query, glm::vec3 target, int depth, bool &right) {
	int d = dimension % depth;
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

void KDTree::KDclosestPoint(std::vector<glm::vec4>& ybufftree, glm::vec3 goal, int &idx) {

	int didx   = 0;
	int depth = 0;
	int top   = 0;
	bool kind = false;
	
	bool good_root = false;

	int bestIdx = 0;
	float bestDist = 1.0f*LONG_MAX;

	std::vector<stack> st(ybufftree.size());
	
	stack_push(st, idx, true, depth, top);
	
	// simluating recursion 
	while (top > 0) {
		//didx, depth, kind, top all by reference
		stack_pop(st, didx, good_root, depth, top);

		if (ybufftree[didx].w == 0.0f){ // null
			break;
		}

		glm::vec3 currPoint(ybufftree[didx].x, ybufftree[didx].y, ybufftree[didx].z);
		bool right = false;
		float dist = compute_distance(goal, currPoint, depth, right);
				
		if (good_root == false && bestDist < dist ) {
			continue;
		}
		
		if (bestDist > dist) {
			bestDist = dist;
			bestIdx = didx;
		}

		int depth_children = depth + 1;
		if (right == true) { // goodSide is Right, // Badside is left
			//bad child == left
			stack_push(st, 2*idx+1, false, depth_children, top);

			//good child == right
			stack_push(st, 2*idx+2, true, depth_children, top);
		}
		else {
			//bad child == right
			stack_push(st, 2 * idx + 2, false, depth_children, top);

			//good child == left
			stack_push(st, 2 * idx + 1, true, depth_children, top);
		}
	}

	
}



void KDTree::initCpuKDTree(std::vector<glm::vec3> &ybuff, std::vector<glm::vec4>&ybufftree) {

	// Inits
	int n = ybuff.size();
	int parentIdx = 0;
	int depth = 0;
	
	std::cout << "Pritnting input \n";
	printKDTree3(ybuff);

	// Compute the KDTree
	buildTree(ybuff, 0, n-1, parentIdx, ybufftree, depth);

	std::cout << "Pritnting level order kd tree \n";
	printKDTree4(ybufftree);


	std::cout << "Searching the kd tree for point \n";
	int idx = -1;
	glm::vec3 goal(7.1, 1.1, 4.1);
	KDclosestPoint(ybufftree, goal, idx);
	
	std::cout << "Closest Point at idx "<<idx<<std::endl;
	std::cout << ybufftree[idx].x<<" " << ybufftree[idx].y<< " " << ybufftree[idx].z << std::endl;


	return;
}