/**
* @file      main.cpp
* @brief     Example Points flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"


// ================
// Configuration
// ================

// LOOK-2.1 LOOK-2.3 - toggles for UNIFORM_GRID and COHERENT_GRID
#define VISUALIZE 1
#define GPUNAIVE 0
#define GPUKDTREE 1
#define COHERENT_GRID 0
#define dims 3

// LOOK-1.2 - change this to adjust particle count in the simulation
int N_FOR_VIS = 5000; // number of points // reset after reading file
float DT = 0.2f;

std::vector<glm::vec3> Ybuffer;
std::vector<glm::vec3> Xbuffer;

glm::vec4 *YbufferTree;
glm::ivec3 *track;
int treesize;
int tracksize;

// Data read function 
void read_data(std::vector<glm::vec3> &buffer, std::string filename, float offset) {

	std::ifstream filein(filename);
	int count = 0;
	int vertices_cnt;
	std::vector<std::string> header_info;

	// ply files have vertex count on line 21
	for (std::string line; std::getline(filein, line); )
	{	// get number of vertices
		if (count == 17) {
			std::istringstream iss(line);
			for (std::string s; iss >> s; )
				header_info.push_back(s);
			vertices_cnt = std::stoi(header_info[2]);
			break;
		}
		count++;
	}
	std::cout << "vertex count :" << vertices_cnt << std::endl;
	filein.clear();
	filein.seekg(0, std::ios::beg);
	count = 0;

	//vertices are stored from line 24 onwards until vcount
	for (std::string line; std::getline(filein, line); )
	{
		if (count > 24 + vertices_cnt) break;

		if (count >= 24) {
			std::istringstream is(line);

			float x=0.0f, y = 0.0f, z = 0.0f;
			is >> x >> y >> z;

			glm::vec3 tmp(x+offset, y+offset, z+offset);
			//std::cout<<x<<" "<<y<<" "<<z<<std::endl;
			buffer.push_back(tmp);

			//int idx = (count - 24)*dims;
			//is >> buffer[idx] >> buffer[idx + 1] >> buffer[idx + 2];
			//cout << buffer[idx] << " " << buffer[idx + 1] << " " << buffer[idx + 2] << endl;
		}
		count++;
	}
	std::cout << "data load completed :" << (count - 24 - 1) << std::endl;
	return;
}

/*
*C main function.
*/
int main(int argc, char* argv[]) {

	projectName = "Project4 PICP";

	// Load data into cpu buffers
	printf("** Read Point Cloud Data **\n");

	std::cout << "Data File Y(target): " << argv[1] << std::endl;
	read_data(Ybuffer, argv[1], 0.0f);

	std::cout << "Data File X(source): " << argv[2] << std::endl;
	read_data(Xbuffer, argv[2], 0.05f);

	// Initialize drawing state
	N_FOR_VIS = Ybuffer.size() + Xbuffer.size();

	std::cout << Ybuffer[0].x << " " << Ybuffer[0].y << " " << Ybuffer[0].z << std::endl;
	std::cout << Xbuffer[0].x << " " << Xbuffer[0].y << " " << Xbuffer[0].z << std::endl;
	std::cout << "total points = " << N_FOR_VIS << std::endl;

#if GPUKDTREE
	std::cout << "Building KDsearch tree for Y" << std::endl;

	//std::vector<glm::vec3> ybuff ={
	//						glm::vec3(1,7,5),
	//						glm::vec3(2,6,6),
	//						glm::vec3(3,5,7),
	//						glm::vec3(4,4,1),
	//						glm::vec3(5,3,2),
	//						glm::vec3(6,2,3),
	//						glm::vec3(7,1,4)
	//						};

	int size = KDTree::nextPowerOf2(2*(Ybuffer.size()+1)); // to store nulls for leaf nodes
	std::vector<glm::vec4> YbuffTree(size, glm::vec4(0.0f));

	// Init mystack
	int sz = (int)log2(size);
	int X = (int)Xbuffer.size();
	std::vector<glm::ivec3> tk(X*sz, glm::ivec3(0,0,0));
	track = &tk[0];

	KDTree::initCpuKDTree(Ybuffer,YbuffTree);
	YbufferTree= &YbuffTree[0];

	tracksize = X * sz;
	treesize  = size;

# endif
 
	if (init(argc, argv)) {
		mainLoop();
		Points::endSimulation();
		return 0;
	}
	else {
		return 1;
	}


	return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {

	// Set window title to "Student Name: [SM 2.0] GPU Name"

	cudaDeviceProp deviceProp;
	int gpuDevice = 0;
	int device_count = 0;

	cudaGetDeviceCount(&device_count);

	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}
	cudaGetDeviceProperties(&deviceProp, gpuDevice);
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	std::ostringstream ss;
	ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
	deviceName = ss.str();

	// Window setup stuff
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		std::cout
			<< "Error: Could not initialize GLFW!"
			<< " Perhaps OpenGL 3.3 isn't available?"
			<< std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	initVAO();

	// Default to device ID 0. If you have more than one GPU and want to test a non-default one,
	// change the device ID.
	cudaGLSetGLDevice(0);

	cudaGLRegisterBufferObject(pointVBO_positions);
	cudaGLRegisterBufferObject(pointVBO_velocities);

	Points::initCpuICP(Ybuffer, Xbuffer);
#if GPUKDTREE
	Points::initGPUKD(Ybuffer, Xbuffer, YbufferTree, track, treesize, tracksize);
#elif GPUNAIVE
	Points::initGPU(Ybuffer, Xbuffer);
#endif

	updateCamera();

	initShaders(program);

	glEnable(GL_DEPTH_TEST);

	return true;
}

void initVAO() {

	std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * (N_FOR_VIS)] };
	std::unique_ptr<GLuint[]> bindices{ new GLuint[N_FOR_VIS] };

	glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
	glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

	for (int i = 0; i < N_FOR_VIS; i++) {
		bodies[4 * i + 0] = 0.0f;
		bodies[4 * i + 1] = 0.0f;
		bodies[4 * i + 2] = 0.0f;
		bodies[4 * i + 3] = 1.0f;
		bindices[i] = i;
	}


	glGenVertexArrays(1, &pointVAO); // Attach everything needed to draw a particle to this
	glGenBuffers(1, &pointVBO_positions);
	glGenBuffers(1, &pointVBO_velocities);
	glGenBuffers(1, &pointIBO);

	glBindVertexArray(pointVAO);

	// Bind the positions array to the pointVAO by way of the pointVBO_positions
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_positions); // bind the buffer
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

	glEnableVertexAttribArray(positionLocation);
	glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Bind the velocities array to the pointVAO by way of the pointVBO_velocities
	glBindBuffer(GL_ARRAY_BUFFER, pointVBO_velocities);
	glBufferData(GL_ARRAY_BUFFER, 4 * (N_FOR_VIS) * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(velocitiesLocation);
	glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, pointIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, (N_FOR_VIS) * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void initShaders(GLuint * program) {
	GLint location;

	program[PROG_BOID] = glslUtility::createProgram(
		"shaders/point.vert.glsl",
		"shaders/point.geom.glsl",
		"shaders/point.frag.glsl", attributeLocations, 2);
	glUseProgram(program[PROG_BOID]);

	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
		glUniform3fv(location, 1, &cameraPosition[0]);
	}
}

//====================================
// Main loop
//====================================

void runCUDA() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
	// use this buffer

	float4 *dptr = NULL;
	float *dptrVertPositions = NULL;
	float *dptrVertVelocities = NULL;

	cudaGLMapBufferObject((void**)&dptrVertPositions, pointVBO_positions);
	cudaGLMapBufferObject((void**)&dptrVertVelocities, pointVBO_velocities);

	// execute the kernel
#if GPUKDTREE
	Points::stepSimulationGPUKD(Ybuffer, Xbuffer, treesize, tracksize , DT);
#elif GPUNAIVE
	Points::stepSimulationGPUNaive(Ybuffer, Xbuffer, DT);
#else
	Points::stepSimulationICPNaive(Ybuffer, Xbuffer, DT);
#endif

#if VISUALIZE
	Points::copyPointsToVBO(dptrVertPositions, dptrVertVelocities);
#endif
	// unmap buffer object
	cudaGLUnmapBufferObject(pointVBO_positions);
	cudaGLUnmapBufferObject(pointVBO_velocities);
}

void mainLoop() {
	double fps = 0;
	double timebase = 0;
	int frame = 0;

	Points::unitTest(); // LOOK-1.2 We run some basic example code to make sure
					   // your CUDA development setup is ready to go.

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		frame++;
		double time = glfwGetTime();

		if (time - timebase > 1.0) {
			fps = frame / (time - timebase);
			timebase = time;
			frame = 0;
		}

		runCUDA();

		std::ostringstream ss;
		ss << "[";
		ss.precision(1);
		ss << std::fixed << fps;
		ss << " fps] " << deviceName;
		glfwSetWindowTitle(window, ss.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if VISUALIZE
		glUseProgram(program[PROG_BOID]);
		glBindVertexArray(pointVAO);
		glPointSize((GLfloat)pointSize);
		glDrawElements(GL_POINTS, N_FOR_VIS + 1, GL_UNSIGNED_INT, 0);
		glPointSize(1.0f);

		glUseProgram(0);
		glBindVertexArray(0);

		glfwSwapBuffers(window);
#endif
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (leftMousePressed) {
		// compute new camera parameters
		phi += (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
		updateCamera();
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
		updateCamera();
	}

	lastX = xpos;
	lastY = ypos;
}

void updateCamera() {
	cameraPosition.x = zoom * sin(phi) * sin(theta);
	cameraPosition.z = zoom * cos(theta);
	cameraPosition.y = zoom * cos(phi) * sin(theta);
	cameraPosition += lookAt;

	projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
	glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
	projection = projection * view;

	GLint location;

	glUseProgram(program[PROG_BOID]);
	if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
		glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
	}
}
