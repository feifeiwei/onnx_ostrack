cmake_minimum_required(VERSION 3.0)
project(ObjectTracking)

set(CMAKE_CXX_STANDARD 11)

# Find OpenCV

set(OpenCV_DIR /home/mengfei_wei/work/tools/opencv4/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add onnxruntime


# Add your source files
set(SOURCES
    main.cpp
)

# Executable
add_executable(run_tracking ${SOURCES})

# Link libraries
target_link_libraries(run_tracking ${OpenCV_LIBS} )
