cmake_minimum_required(VERSION 3.1)

project(cis565_PICP)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
list(APPEND CUDA_NVCC_FLAGS_RELWITHDEBUGINFO -lineinfo)

set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(headers
    src/cudaMat4.hpp
    src/glslUtility.hpp
    src/kernel.h
    src/main.hpp
    src/utilityCore.hpp
    src/svd3.h
    src/kdtree.hpp
    src/common.h
    )

set(sources
    src/glslUtility.cpp
    src/kernel.cu
    src/main.cpp
    src/utilityCore.cpp
    src/kdtree.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBRARIES})

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )

##################################################################

# Crucial magic for CUDA linking
#find_package(Threads REQUIRED)
#find_package(CUDA 10 REQUIRED)

#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#set(CUDA_SEPARABLE_COMPILATION ON)

#if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#endif()

#include_directories(.)
#add_subdirectory(PICP)

#cuda_add_executable(${CMAKE_PROJECT_NAME}
#    "src/main.cpp"
#	"src/testing_helpers.hpp"
#    )

#target_link_libraries(${CMAKE_PROJECT_NAME}
#	PICP
#    ${CORELIBS}
#    )
