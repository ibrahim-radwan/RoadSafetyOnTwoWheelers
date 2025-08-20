#include "radarheatmapanalyser.hpp"
#include "config.hpp"
#include "exceptions.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <thread>

namespace radar {

RadarHeatmapAnalyser::RadarHeatmapAnalyser(const std::string& config_file)
    : range_fft_plan_(nullptr), 
      doppler_fft_plan_(nullptr),
      fftw_input_(nullptr),
      fftw_output_(nullptr),
      is_indoor_(true),
      angle_range_(90),
      angle_resolution_(1),
      angle_bins_(angle_range_ / angle_resolution_)
{
    if (!config_file.empty()) {
        initialize(config_file);
    }
}

RadarHeatmapAnalyser::~RadarHeatmapAnalyser() {
    cleanupFFTW();
}

bool RadarHeatmapAnalyser::initialize(const std::string& config_file) {
    try {
        config_file_path_ = config_file;
        
        // Load radar configuration
        // Directly construct AdcParams from config file; includes derived fields
        adc_params_ = std::make_shared<AdcParams>(config_file);
        
        // Initialize ArrayFire (use CPU backend by default, can be changed to GPU)
        af::setBackend(AF_BACKEND_CPU);
        af::info(); // Print ArrayFire info
        
        // Initialize FFTW plans
        initializeFFTWPlans(adc_params_->samples, adc_params_->chirps);
        
        is_initialized_ = true;
        
        std::cout << "RadarHeatmapAnalyser initialized successfully:" << std::endl;
        std::cout << "  Config file: " << config_file << std::endl;
        std::cout << "  TX antennas: " << adc_params_->tx << std::endl;
        std::cout << "  RX antennas: " << adc_params_->rx << std::endl;
        std::cout << "  Samples: " << adc_params_->samples << std::endl;
        std::cout << "  Chirps: " << adc_params_->chirps << std::endl;
        std::cout << "  Range resolution: " << std::fixed << std::setprecision(4)
                  << adc_params_->range_resolution << " m" << std::endl;
        std::cout << "  Doppler resolution: " << std::fixed << std::setprecision(4)
                  << adc_params_->doppler_resolution << " m/s" << std::endl;
        // Derived resolutions are not present in AdcParams; skip printing them for now
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to initialize RadarHeatmapAnalyser: " << e.what() << std::endl;
        is_initialized_ = false;
        return false;
    }
}

AnalysisResult RadarHeatmapAnalyser::analyseFrame(const std::shared_ptr<RadarFrame>& frame) {
    if (!is_initialized_) {
        throw RadarException("RadarHeatmapAnalyser not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocess frame to complex format
    af::array complex_frame = preprocessFrameFromRawData(frame->getRawData());
    
    // Process frame (stub implementation)
    AnalysisResult result = processFrameStub(complex_frame);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0;
    
    // Set frame metadata
    result.frame_timestamp = frame->getTimestamp();
    result.frame_number = frame->getFrameNumber();
    
    // Log performance stats periodically
    if (frame_count_.load() % 10 == 0) {
        logPerformanceStats(result.processing_time_ms);
    }
    frame_count_++;
    
    return result;
}

void RadarHeatmapAnalyser::run(ThreadSafeQueue<std::shared_ptr<RadarFrame>>& input_queue,
                               ThreadSafeQueue<AnalysisResult>& output_queue,
                               std::atomic<bool>& stop_flag) {
    if (!is_initialized_) {
        std::cerr << "RadarHeatmapAnalyser not initialized, cannot start processing" << std::endl;
        return;
    }

    std::cout << "RadarHeatmapAnalyser processing thread started" << std::endl;

    while (!stop_flag.load()) {
        try {
            std::shared_ptr<RadarFrame> frame_ptr;
            if (input_queue.tryPop(frame_ptr)) {
                AnalysisResult result = analyseFrame(frame_ptr);
                output_queue.push(std::move(result));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in RadarHeatmapAnalyser processing (shared_ptr): " << e.what() << std::endl;
        }
    }

    std::cout << "RadarHeatmapAnalyser processing thread stopped" << std::endl;
}

af::array RadarHeatmapAnalyser::preprocessFrameFromRawData(const RawDataVector& raw_data) {
    // For now, create a zero-initialized complex frame with expected dimensions.
    // This avoids assumptions about raw storage order and keeps the stub working.
    if (!adc_params_) {
        throw RadarException("ADC parameters not available for preprocessing");
    }

    const size_t expected_size = static_cast<size_t>(adc_params_->chirps) *
                                 static_cast<size_t>(adc_params_->tx) *
                                 static_cast<size_t>(adc_params_->samples) *
                                 2ULL * // I & Q
                                 static_cast<size_t>(adc_params_->rx);

    if (raw_data.size() != expected_size) {
        // Log mismatch but still proceed with zeroed data for robustness in stub
        std::cerr << "Warning: Raw data size mismatch. Expected: "
                  << expected_size << ", Got: " << raw_data.size() << std::endl;
    }

    af::array complex_frame = af::constant(0, 
        adc_params_->chirps,
        adc_params_->tx,
        adc_params_->rx,
        adc_params_->samples,
        c32);

    return complex_frame;
}

AnalysisResult RadarHeatmapAnalyser::processFrameStub(const af::array& complex_frame) {
    // Stub implementation that returns correctly structured empty results
    // This is where the actual openradar_pd_process_frame logic would go
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Create default result with correct dimensions
    AnalysisResult result = createDefaultResult(0.0, 0, 0.0);
    
    // Set correct dimensions based on ADC parameters
    result.range_bins = adc_params_->samples;
    result.doppler_bins = adc_params_->chirps;
    result.azimuth_bins = angle_bins_;
    
    // Create empty heatmaps with correct dimensions
    // Range-Doppler heatmap: range_bins x doppler_bins (set to None/empty for openradar method)
    result.range_doppler.clear(); // OpenRadar method doesn't compute range-doppler
    
    // Range-Azimuth heatmap: angle_bins x range_bins
    result.range_azimuth.resize(angle_bins_, std::vector<double>(adc_params_->samples, 0.0));
    
    // Empty point cloud data (no detections in stub)
    result.point_cloud.clear();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0;
    
    return result;
}

std::vector<std::vector<double>> RadarHeatmapAnalyser::arrayFireToVector2D(const af::array& af_array) {
    // Convert ArrayFire array to 2D vector
    if (af_array.numdims() != 2) {
        throw RadarException("arrayFireToVector2D expects 2D array");
    }
    
    af::dim4 dims = af_array.dims();
    size_t rows = dims[0];
    size_t cols = dims[1];
    
    // Get host data
    std::vector<double> host_data(af_array.elements());
    af_array.host(host_data.data());
    
    // Convert to 2D vector (row-major)
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = host_data[i * cols + j];
        }
    }
    
    return result;
}

void RadarHeatmapAnalyser::initializeFFTWPlans(size_t samples, size_t chirps) {
    // Allocate FFTW memory
    fftw_input_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * samples);
    fftw_output_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * samples);
    
    if (!fftw_input_ || !fftw_output_) {
        throw RadarException("Failed to allocate FFTW memory");
    }
    
    // Create FFTW plans
    range_fft_plan_ = fftw_plan_dft_1d(samples, fftw_input_, fftw_output_, 
                                      FFTW_FORWARD, FFTW_MEASURE);
    
    // Doppler FFT plan (for future use)
    doppler_fft_plan_ = fftw_plan_dft_1d(chirps, fftw_input_, fftw_output_,
                                        FFTW_FORWARD, FFTW_MEASURE);
    
    if (!range_fft_plan_ || !doppler_fft_plan_) {
        throw RadarException("Failed to create FFTW plans");
    }
    
    std::cout << "FFTW plans initialized for " << samples << " range samples and " 
              << chirps << " chirps" << std::endl;
}

void RadarHeatmapAnalyser::cleanupFFTW() {
    if (range_fft_plan_) {
        fftw_destroy_plan(range_fft_plan_);
        range_fft_plan_ = nullptr;
    }
    
    if (doppler_fft_plan_) {
        fftw_destroy_plan(doppler_fft_plan_);
        doppler_fft_plan_ = nullptr;
    }
    
    if (fftw_input_) {
        fftw_free(fftw_input_);
        fftw_input_ = nullptr;
    }
    
    if (fftw_output_) {
        fftw_free(fftw_output_);
        fftw_output_ = nullptr;
    }
    
    fftw_cleanup();
}

void RadarHeatmapAnalyser::logPerformanceStats(double processing_time) const {
    std::cout << "[RadarHeatmapAnalyser] Frame " << frame_count_.load() 
              << " processed in " << std::fixed << std::setprecision(2) 
              << processing_time << " ms" << std::endl;
}

AnalysisResult RadarHeatmapAnalyser::createDefaultResult(double frame_timestamp,
                                                       size_t frame_number,
                                                       double processing_time) const {
    AnalysisResult result;
    
    result.frame_timestamp = frame_timestamp;
    result.frame_number = frame_number;
    result.processing_time_ms = processing_time;
    
    // Set dimensions
    if (adc_params_) {
        result.range_bins = adc_params_->samples;
        result.doppler_bins = adc_params_->chirps;
    }
    result.azimuth_bins = angle_bins_;
    
    return result;
}

std::string RadarHeatmapAnalyser::toString() const {
    std::ostringstream oss;
    oss << "RadarHeatmapAnalyser{"
        << "initialized=" << is_initialized_.load()
        << ", config_file=" << config_file_path_
        << ", is_indoor=" << is_indoor_
        << ", angle_range=" << angle_range_
        << ", angle_resolution=" << angle_resolution_
        << ", frames_processed=" << frame_count_.load();
    
    if (adc_params_) {
        oss << ", tx=" << adc_params_->tx
            << ", rx=" << adc_params_->rx
            << ", samples=" << adc_params_->samples
            << ", chirps=" << adc_params_->chirps;
    }
    
    oss << "}";
    return oss.str();
}

} // namespace radar
