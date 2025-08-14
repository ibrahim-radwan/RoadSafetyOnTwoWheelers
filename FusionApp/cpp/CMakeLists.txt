cmake_minimum_required(VERSION 3.10)

# Project name
project(FusionReplay)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find ArrayFire
find_package(ArrayFire QUIET)
if(ArrayFire_FOUND)
    message(STATUS "ArrayFire found")
    set(ARRAYFIRE_ENABLED ON)
else()
    message(WARNING "ArrayFire not found. Radar analysis features will be disabled.")
    set(ARRAYFIRE_ENABLED OFF)
endif()

# Find FFTW3
pkg_check_modules(FFTW3 QUIET fftw3)
if(FFTW3_FOUND)
    message(STATUS "FFTW3 found")
    set(FFTW_ENABLED ON)
else()
    message(WARNING "FFTW3 not found. FFT operations will use fallback implementation.")
    set(FFTW_ENABLED OFF)
endif()

# Base radar libraries
set(RADAR_SOURCES
    radar/config.cpp
    radar/syncstate.cpp
    radar/radardata.cpp
    radar/recordingfeed.cpp
)

# Add radar analysis sources if dependencies are available
if(ARRAYFIRE_ENABLED AND FFTW_ENABLED)
    list(APPEND RADAR_SOURCES
        radar/radarheatmapanalyser.cpp
    )
    message(STATUS "Radar analysis with ArrayFire + FFTW enabled")
    add_definitions(-DRADAR_ANALYSIS_ENABLED)
endif()

# Add executable
add_executable(fusion_replay 
    fusion_replay.cpp 
    ${RADAR_SOURCES}
)

# Link libraries
target_link_libraries(fusion_replay)

# Add ArrayFire if available
if(ARRAYFIRE_ENABLED AND FFTW_ENABLED)
    target_link_libraries(fusion_replay ArrayFire::afcpu)  # Use CPU backend by default
    target_link_libraries(fusion_replay ${FFTW3_LIBRARIES})
    target_include_directories(fusion_replay PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_compile_options(fusion_replay PRIVATE ${FFTW3_CFLAGS_OTHER})
endif()

# Create test executable for radar analysis (if enabled)
if(ARRAYFIRE_ENABLED AND FFTW_ENABLED)
    add_executable(test_radar_analysis
        test_radar_analysis.cpp
        ${RADAR_SOURCES}
    )
    
    target_link_libraries(test_radar_analysis ArrayFire::afcpu ${FFTW3_LIBRARIES})
    target_include_directories(test_radar_analysis PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_compile_options(test_radar_analysis PRIVATE ${FFTW3_CFLAGS_OTHER})
endif()
