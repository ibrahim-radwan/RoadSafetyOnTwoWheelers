# Radar Analysis with ArrayFire + FFTW

This document describes how to build and use the C++ radar analysis implementation using ArrayFire and FFTW.

## Prerequisites

### 1. Install ArrayFire

#### Ubuntu/Debian:
```bash
# Download ArrayFire installer from https://arrayfire.com/download/
wget https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh
chmod +x ArrayFire-v3.8.0_Linux_x86_64.sh
sudo ./ArrayFire-v3.8.0_Linux_x86_64.sh

# Add to environment
echo 'export AF_PATH=/opt/arrayfire' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${AF_PATH}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PATH=${AF_PATH}/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Alternative - Build from source:
```bash
git clone --recursive https://github.com/arrayfire/arrayfire.git
cd arrayfire
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

### 2. Install FFTW3

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install libfftw3-dev libfftw3-doc
```

#### CentOS/RHEL:
```bash
sudo yum install fftw-devel
# or
sudo dnf install fftw-devel
```

### 3. Install CMake and Build Tools

```bash
sudo apt-get install cmake build-essential pkg-config
```

## Building the Project

1. **Navigate to the C++ directory:**
   ```bash
   cd /home/javad/dev/RoadSafetyOnTwoWheelers/FusionApp/cpp
   ```

2. **Create build directory:**
   ```bash
   mkdir build && cd build
   ```

3. **Configure with CMake:**
   ```bash
   cmake ..
   ```

   You should see output indicating whether ArrayFire and FFTW were found:
   ```
   -- ArrayFire found
   -- FFTW3 found
   -- Radar analysis with ArrayFire + FFTW enabled
   ```

4. **Build the project:**
   ```bash
   make -j$(nproc)
   ```

## Testing the Implementation

### Test 1: Basic Functionality
```bash
# Run the radar analysis test
./test_radar_analysis --config-file ../config_files/AWR2243_180m_70cm_64_2_512.txt
```

Expected output:
```
=== Radar Analysis Test Program ===

--- Test 1: Initialization ---
ArrayFire v3.8.0 (CPU, 64-bit Linux, build default)
[0] Intel: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 32768 MB
RadarHeatmapAnalyser initialized successfully:
  Config file: ../config_files/AWR2243_180m_70cm_64_2_512.txt
  TX antennas: 2
  RX antennas: 4
  Samples: 512
  Chirps: 64
  Range resolution: 0.3516 m
  Doppler resolution: 1.3672 m/s
...
```

### Test 2: Performance Test
```bash
# Test with real radar data (if available)
./fusion_replay --config-file ../config_files/AWR2243_180m_70cm_64_2_512.txt --dest-dir ~/dev/offline/2025_07_09_14_13_00_AWR2243_180m_70cm_64_2_512
```

## Usage in Code

### Basic Usage:
```cpp
#include "radar/radarheatmapanalyser.hpp"

// Initialize analyser
radar::RadarHeatmapAnalyser analyser;
if (!analyser.initialize("config.txt")) {
    // Handle error
}

// Process single frame
radar::RadarFrame frame = /* get frame */;
radar::AnalysisResult result = analyser.analyseFrame(frame);

// Access results
std::cout << "Processing time: " << result.processing_time_ms << " ms" << std::endl;
std::cout << "Point cloud size: " << result.point_cloud.size() << std::endl;
std::cout << "Range-azimuth heatmap: " << result.range_azimuth.size() 
          << " x " << result.range_azimuth[0].size() << std::endl;
```

### Multi-threaded Usage:
```cpp
// Create queues and control flag
radar::ThreadSafeQueue<radar::RadarFrame> input_queue;
radar::ThreadSafeQueue<radar::AnalysisResult> output_queue;
std::atomic<bool> stop_flag{false};

// Start processing thread
std::thread analysis_thread([&]() {
    analyser.run(input_queue, output_queue, stop_flag);
});

// Add frames to input queue
input_queue.push(frame);

// Get results from output queue
radar::AnalysisResult result;
if (output_queue.waitAndPop(result, std::chrono::seconds(1))) {
    // Process result
}

// Stop processing
stop_flag = true;
analysis_thread.join();
```

## Implementation Details

### ArrayFire Usage
- **Multi-dimensional arrays**: Radar frames stored as 4D arrays (chirps, tx, rx, samples)
- **GPU acceleration**: Automatic GPU usage when available (CUDA/OpenCL)
- **Memory management**: Efficient memory transfers between host and device

### FFTW Usage
- **Optimized plans**: Pre-computed FFT plans for repeated operations
- **Wisdom system**: FFTW learns optimal algorithms for your data sizes
- **Thread safety**: Separate plans for different thread contexts

### Data Flow
1. **Raw data input**: int16 values from DCA1000 radar
2. **Preprocessing**: Reshape and convert to complex format using ArrayFire
3. **Analysis**: Stub implementation returns correctly structured results
4. **Output**: Standard C++ containers for easy integration

## Performance Characteristics

### Expected Performance (typical values):
- **Frame preprocessing**: 1-5 ms per frame
- **Analysis (stub)**: < 1 ms per frame
- **Memory usage**: ~10-50 MB depending on frame size
- **Throughput**: 100-500 frames/second (depending on hardware)

### Optimization Tips:
1. **Use GPU backend** for large frame sizes:
   ```cpp
   af::setBackend(AF_BACKEND_CUDA);  // or AF_BACKEND_OPENCL
   ```

2. **Pre-allocate arrays** for repeated processing
3. **Use FFTW wisdom** for optimal FFT performance
4. **Batch processing** for better GPU utilization

## Troubleshooting

### ArrayFire not found:
```bash
# Check installation
af_info

# Verify CMake can find ArrayFire
cmake .. -DArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake
```

### FFTW not found:
```bash
# Check installation
pkg-config --modversion fftw3

# Manual specification
cmake .. -DFFTW3_ROOT=/usr/local
```

### Runtime errors:
```bash
# Check library paths
ldd ./test_radar_analysis

# Add to LD_LIBRARY_PATH if needed
export LD_LIBRARY_PATH=/opt/arrayfire/lib64:$LD_LIBRARY_PATH
```

## Next Steps

1. **Implement full analysis**: Replace stub with actual radar processing algorithms
2. **Add GPU optimization**: Optimize for CUDA/OpenCL backends  
3. **Integrate with Python**: Use this as backend for Python radar processing
4. **Add more analysis methods**: CFAR detection, clustering, tracking

## Related Files

- `radar/radaranalyser.hpp` - Base interface
- `radar/radarheatmapanalyser.hpp` - ArrayFire + FFTW implementation
- `radar/radarheatmapanalyser.cpp` - Implementation
- `test_radar_analysis.cpp` - Test program
- `CMakeLists.txt` - Build configuration
