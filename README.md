 <p align="center">
  <img src="img/gifForTitle2.gif">
</p>

CUDA Iterative Closest Point
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**


* Author: Chhavi Sharma ([LinkedIn](https://www.linkedin.com/in/chhavi275/))
* Tested on: Windows 10, Intel Core(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, 
             NVIDIA Quadro P1000 4GB (MOORE100B-06)

### Index

- [Introduction]( )
- [Algorithm]( )
- [Implementation Details]( )
- [Analysis]( )
- [Some More Results and Bloopers]( )
- [Resources and References]( )

### Introduction 

In this project, we show different optimzations of the iterative closest point algorithm that aligns patially overlapping point clouds of different views of an object.
Operations in ICP on large point clouds are highly parallelizable which makes it a good candiate for CUDA based implemantion and optimization.

(Iterative closest point algorithm)[https://en.wikipedia.org/wiki/Iterative_closest_point] successively estimates and applies rotation and transaltion between two sets of point clouds of different views of an object to achieve the closest alignment. 
The algorithm iteratively revises the transformation needed to minimize the distance between corresponding points across the two point clouds. ICP depends on an initial guess of the rigid body transformation (Rotation and translation) to acheive good results in case of drastically different views of objects.


### Algorithm
Given:
 - X : N Source points (x_i, y_i, z_i)
 - Y : M Target points (x_i, y_i, z_i)
 
At each iteration:
 - FOr each point in the source, find the closest correspoding point in y based on some metric. We use minimum euclidian distance to assign the closest point.
 - Now, for a set of correspondances, we estimation the rotation and transaltion between them by solving the (orthogonal procrustes problem)[https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem]. The steps involved solve the following optmization problem are as follows:
         -![]()
         -![]()
         - We do this by mean centring the source and target corrspondances, and then computing the matrix W= transpose(Ymeancntred)* Xmeancntred.
         - Then, the Rotation is U * Transpose(V) where singualr value decomposition of W, ie.e SVD(W) = USV. 
         - Translation,T is Ymean - R * Xmean.
 - Reapeat until convergence i.e. when predicted Rotation matrix is identity and translation is close to zero.
       

### Implementation Details

Three variations of ICP on have been implmented:
   - [x] CPU Iterative Closest Point  
   - [x] GPU Iterative Closest Point with Naive Search
   - [x] GPU Iterative Closest Point with KDTree Search

#### CPU Iterative Closest Point  
The CPU implementation uses the steps in the above algorithm to iteratively apply roation andtranlation on source data. The correnpondance computation requires an O(M) search for each element in the source point cloud. This is done naively on th CPu where each source points looks through the entire traget set to compute the closest correpondence. 

#### GPU Iterative Closest Point with Naive Search
The CPU implementation is optimised by using a CUDA kernel to perfrom the coresspondance search. Each element in the source point cloud now finds a correspondance in the target point cloud in parallel. Even though this approach is much faster than the CPU version, each point still goes through the entire target data set to pick the closest point.

#### GPU Iterative Closest Point with KDTree Search
We further optimize each iteration by optimizing the search per source point. We implement a KD-tree structure to search the target 3D point cloud data. A kd tree is a binary tree in which every leaf node is a k-dimensional point. Every non-leaf node can be thought of as implicitly generating a splitting hyperplane that divides the space into two parts, known as half-spaces. Points to the left of this hyperplane are represented by the left subtree of that node and points to the right of the hyperplane are represented by the right subtree. 
![]()
![]()
The average search time on a 3D tree for target data of size n is O(log(n)).
The K-D tree is constructed on the CPU and the stored in a contiguous linear level-order traveral format. It is then transfered to the GPU where the search travel is iterative rather than recursive(CUDA recursion limitations). To facilitate iterative traveral and backtracking over the tree, a book-keeping array isalso maintained. 

#### Iterative Rendering
The point cloud data is also rendered iteratively to show the chages made by the application of each rotation and translation predicted by the algorithm. 


### Analysis

The time taken per iteration for the above three cases is plotted below:-
We cansee that only in the K-D tree seach, the time taken per iteration redues as the point cloud aligns better with the target.
The initial iterations of GPU ICP with KDtree is slower than the GPU Naive search eve nthough the number of points being loooked at per point during the searc are much lesser in case of KDtree search. This may be becuase of GPU memory overheads. The naive search accesses more data but it is contiguos, whereas, KD tree search jumps around nodes( this can't be predetermined and therefore stored contiguosly) looking at non-contiguous data and thus takes more time even with lesser comaprasions. 
![]()

The average Number of nodes explored per KD-tree seach is also shown below. As the itertions increase and the points clouds aligh better, lesser nodes are searched for in the tree and the runtime decreases.
![]()


### More Rsults and Bloopers

More Results:
![COW](img/cow.gif)
![ANT](img/ant.gif)

Error in the Rotation computation deformed the point cloud:
![](img/blopper.gif)

### Resources and References 
[icp.pdf](http://ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/17-icp.pdf)  
[wiki/K-d_tree](https://en.wikipedia.org/wiki/K-d_tree)  
[k-d-trees](https://blog.krum.io/k-d-trees/)   
[cmu.kdtrees.pdf](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf)  
[cmu.kdrangenn](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdrangenn.pdf)  
[gfg.k-dimensional-tree](https://www.geeksforgeeks.org/k-dimensional-tree/)  
[GTC-2010/pdfs/2140_GTC2010.pdf](https://www.nvidia.com/content/GTC-2010/pdfs/2140_GTC2010.pdf)  
[s10766-018-0571-0](https://link.springer.com/article/10.1007/s10766-018-0571-0)  
[nghiaho.com](https://nghiaho.com/?p=437)  

