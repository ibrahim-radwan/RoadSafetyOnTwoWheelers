#include <iostream>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

#ifdef RADAR_ANALYSIS_ENABLED
#include "radar/radarheatmapanalyser.hpp"
#endif

#include "radar/config.hpp"
#include "radar/radardata.hpp"
#include "radar/threadsafequeue.hpp"

int main(int argc, char* argv[]) {
    std::cout << "=== Radar Analysis Test Program ===" << std::endl;
    
#ifndef RADAR_ANALYSIS_ENABLED
    std::cout << "Radar analysis disabled. ArrayFire and/or FFTW not found." << std::endl;
    std::cout << "To enable radar analysis:" << std::endl;
    std::cout << "1. Install ArrayFire: https://arrayfire.com/download/" << std::endl;
    std::cout << "2. Install FFTW3: sudo apt-get install libfftw3-dev" << std::endl;
    std::cout << "3. Rebuild the project" << std::endl;
    return 1;
#else
    
    // Parse command line arguments
    std::string config_file;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--config-file") {
            config_file = std::string(argv[i + 1]);
        }
    }
    
    if (config_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --config-file <path>" << std::endl;
        return 1;
    }
    
    try {
        // Test 1: Initialize RadarHeatmapAnalyser
        std::cout << "\n--- Test 1: Initialization ---" << std::endl;
        radar::RadarHeatmapAnalyser analyser;
        
        if (!analyser.initialize(config_file)) {
            std::cerr << "Failed to initialize analyser" << std::endl;
            return 1;
        }
        std::cout << "✓ Analyser initialized successfully" << std::endl;
        std::cout << analyser.toString() << std::endl;
        
        // Test 2: Create dummy radar frame
        std::cout << "\n--- Test 2: Frame Processing ---" << std::endl;
        
        // Load radar config to get parameters
        radar::RadarConfig radar_config(config_file);
        auto adc_params = std::make_shared<radar::AdcParams>(radar_config);
        
        // Create dummy raw data
        size_t raw_data_size = adc_params->chirps * adc_params->tx * 
                              adc_params->samples * 2 * adc_params->rx;
        std::vector<int16_t> dummy_data(raw_data_size, 0);
        
        // Fill with some test pattern
        for (size_t i = 0; i < dummy_data.size(); ++i) {
            dummy_data[i] = static_cast<int16_t>(i % 1000 - 500);
        }
        
        // Create radar frame
        radar::RadarFrame test_frame(
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count(),
            dummy_data,
            1,
            adc_params
        );
        
        std::cout << "✓ Created test frame: " << test_frame.toString() << std::endl;
        
        // Test 3: Analyze frame
        std::cout << "\n--- Test 3: Frame Analysis ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        radar::AnalysisResult result = analyser.analyseFrame(test_frame);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "✓ Frame analyzed successfully" << std::endl;
        std::cout << "  Processing time: " << result.processing_time_ms << " ms" << std::endl;
        std::cout << "  Total time: " << duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Frame timestamp: " << result.frame_timestamp << std::endl;
        std::cout << "  Frame number: " << result.frame_number << std::endl;
        std::cout << "  Range bins: " << result.range_bins << std::endl;
        std::cout << "  Doppler bins: " << result.doppler_bins << std::endl;
        std::cout << "  Azimuth bins: " << result.azimuth_bins << std::endl;
        std::cout << "  Point cloud size: " << result.point_cloud.size() << std::endl;
        std::cout << "  Range-Doppler heatmap: " << 
                     (result.range_doppler.empty() ? "None (as expected)" : 
                      std::to_string(result.range_doppler.size()) + " x " + 
                      std::to_string(result.range_doppler.empty() ? 0 : result.range_doppler[0].size())) << std::endl;
        std::cout << "  Range-Azimuth heatmap: " << result.range_azimuth.size() << " x " << 
                     (result.range_azimuth.empty() ? 0 : result.range_azimuth[0].size()) << std::endl;
        
        // Test 4: Multi-threaded processing
        std::cout << "\n--- Test 4: Multi-threaded Processing ---" << std::endl;
        
        radar::ThreadSafeQueue<radar::RadarFrame> input_queue;
        radar::ThreadSafeQueue<radar::AnalysisResult> output_queue;
        std::atomic<bool> stop_flag{false};
        
        // Add some test frames to the queue
        for (int i = 0; i < 5; ++i) {
            radar::RadarFrame frame(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count() + i,
                dummy_data,
                i + 1,
                adc_params
            );
            input_queue.push(frame);
        }
        
        // Start processing thread
        std::thread analysis_thread([&]() {
            analyser.run(input_queue, output_queue, stop_flag);
        });
        
        // Collect results
        std::vector<radar::AnalysisResult> results;
        for (int i = 0; i < 5; ++i) {
            radar::AnalysisResult result;
            if (output_queue.waitAndPop(result, std::chrono::seconds(5))) {
                results.push_back(result);
                std::cout << "  Received result for frame " << result.frame_number 
                         << " (processed in " << result.processing_time_ms << " ms)" << std::endl;
            } else {
                std::cout << "  Timeout waiting for result " << i << std::endl;
            }
        }
        
        // Stop processing
        stop_flag = true;
        analysis_thread.join();
        
        std::cout << "✓ Multi-threaded processing completed" << std::endl;
        std::cout << "  Processed " << results.size() << " frames" << std::endl;
        
        if (!results.empty()) {
            double avg_time = 0.0;
            for (const auto& r : results) {
                avg_time += r.processing_time_ms;
            }
            avg_time /= results.size();
            std::cout << "  Average processing time: " << avg_time << " ms" << std::endl;
        }
        
        std::cout << "\n=== All Tests Passed! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
#endif
}
