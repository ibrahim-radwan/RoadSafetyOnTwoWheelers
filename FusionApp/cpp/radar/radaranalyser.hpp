#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include "radardata.hpp"
#include "threadsafequeue.hpp"

namespace radar {

/**
 * Point cloud data structure for radar targets
 */
struct PointCloudData {
    std::vector<double> x;           // X coordinates in meters
    std::vector<double> y;           // Y coordinates in meters  
    std::vector<double> z;           // Z coordinates in meters
    std::vector<double> intensity;   // Signal intensity (SNR in dB)
    
    size_t size() const { return x.size(); }
    void clear() { x.clear(); y.clear(); z.clear(); intensity.clear(); }
    void reserve(size_t capacity) {
        x.reserve(capacity);
        y.reserve(capacity);
        z.reserve(capacity);
        intensity.reserve(capacity);
    }
};

/**
 * Analysis result structure containing heatmaps and point cloud data
 */
struct AnalysisResult {
    // Heatmap data (2D matrices stored as row-major vectors)
    std::vector<std::vector<double>> range_doppler;   // Range-Doppler heatmap
    std::vector<std::vector<double>> range_azimuth;   // Range-Azimuth heatmap
    
    // Point cloud data
    PointCloudData point_cloud;
    
    // Metadata
    double processing_time_ms;      // Processing time in milliseconds
    double frame_timestamp;         // Original frame timestamp
    size_t frame_number;            // Frame sequence number
    
    // Dimensions for heatmap interpretation
    size_t range_bins;              // Number of range bins
    size_t doppler_bins;           // Number of Doppler bins
    size_t azimuth_bins;           // Number of azimuth bins
    
    AnalysisResult() : processing_time_ms(0.0), frame_timestamp(0.0), 
                       frame_number(0), range_bins(0), doppler_bins(0), azimuth_bins(0) {}
};

/**
 * Abstract base class for radar data analysis
 * Follows the same interface pattern as the existing RadarFeed classes
 */
class RadarAnalyser {
public:
    virtual ~RadarAnalyser() = default;

    /**
     * Main analysis method that processes radar frames
     * @param frame Input radar frame to analyze
     * @return Analysis results containing heatmaps and point cloud data
     */
    virtual AnalysisResult analyseFrame(const RadarFrame& frame) = 0;

    /**
     * Main processing loop for threaded operation
     * Reads from input queue, processes frames, and writes to output queue
     * @param input_queue Queue to receive RadarFrame objects from
     * @param output_queue Queue to send AnalysisResult objects to  
     * @param stop_flag Atomic flag to signal when to stop processing
     */
    virtual void run(ThreadSafeQueue<RadarFrame>& input_queue,
                     ThreadSafeQueue<AnalysisResult>& output_queue,
                     std::atomic<bool>& stop_flag) = 0;

    /**
     * Initialize the analyser with configuration
     * @param config_file Path to radar configuration file
     * @return true if initialization successful, false otherwise
     */
    virtual bool initialize(const std::string& config_file) = 0;

    /**
     * Get string representation of the analyser
     */
    virtual std::string toString() const = 0;

protected:
    // Common protected members for derived classes
    std::shared_ptr<AdcParams> adc_params_;
    std::atomic<bool> is_initialized_{false};
    std::string config_file_path_;
};

} // namespace radar
