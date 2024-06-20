cmake_minimum_required(VERSION 3.0)
project(ObjectTracking)

set(CMAKE_CXX_STANDARD 14)

# Find OpenCV

set(OpenCV_DIR /home/brave-ws-05/tools/opencv46_install/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})


# Add trt
set(TENSORRT_DIR /home/brave-ws-05/trt/TensorRT-7.2.3.4)
include_directories(/home/brave-ws-05/trt/TensorRT-7.2.3.4/include)
link_directories(/home/brave-ws-05/trt/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib/)

# Find TensorRT libraries
find_library(TensorRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_DIR}/targets/x86_64-linux-gnu/lib/)
find_library(TensorRT_LIBRARY_ONNX nvonnxparser HINTS ${TENSORRT_DIR}/targets/x86_64-linux-gnu/lib/)
set(TensorRT_LIBRARIES ${TensorRT_LIBRARY_INFER} ${TensorRT_LIBRARY_ONNX})

# Add your source files
set(SOURCES
    main.cpp
)

# Executable
add_executable(run_tracking ${SOURCES})

# Link libraries
target_link_libraries(run_tracking ${OpenCV_LIBS} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES})
