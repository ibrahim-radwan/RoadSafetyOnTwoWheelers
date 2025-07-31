cmake_minimum_required(VERSION 3.10)

# Project name
project(FusionReplay)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(fusion_replay fusion_replay.cpp)

# Link OpenCV libraries
target_link_libraries(fusion_replay ${OpenCV_LIBS})

# Print OpenCV information (optional, for debugging)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
