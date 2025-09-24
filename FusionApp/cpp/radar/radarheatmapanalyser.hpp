#pragma once

#include <arrayfire.h>
#include <fftw3.h>
#include <chrono>
#include <complex>
#include "radaranalyser.hpp"

namespace radar
{

/**
 * Radar heatmap analyser implementation using ArrayFire for multi-dimensional
 * arrays and FFTW for high-performance FFT operations.
 *
 * This class translates the Python openradar_pd_process_frame functionality
 * to C++ using ArrayFire for GPU acceleration and FFTW for optimized FFTs.
 */
class RadarHeatmapAnalyser : public RadarAnalyser
{
  public:
    /**
     * Constructor with optional configuration file
     * @param config_file Path to radar configuration file (optional)
     */
    explicit RadarHeatmapAnalyser(const std::string& config_file = "");

    /**
     * Destructor - cleans up FFTW plans and ArrayFire resources
     */
    ~RadarHeatmapAnalyser();

    // Override base class methods
    AnalysisResult analyseFrame(
        const std::shared_ptr<RadarFrame>& frame) override;

    // Single run variant: consume frames from shared_ptr queue for efficiency
    void run(ThreadSafeQueue<std::shared_ptr<RadarFrame>>& input_queue,
             ThreadSafeQueue<AnalysisResult>& output_queue,
             std::atomic<bool>& stop_flag) override;

    bool initialize(const std::string& config_file) override;

    std::string toString() const override;

  private:
    // ArrayFire arrays for radar data processing
    af::array complex_frame_;      // (chirps, tx, rx, samples) complex data
    af::array range_fft_result_;   // Range FFT results
    af::array range_azimuth_map_;  // Range-azimuth heatmap

    // FFTW plans for optimized FFT operations
    fftw_plan range_fft_plan_;    // Range FFT plan
    fftw_plan doppler_fft_plan_;  // Doppler FFT plan (for future use)

    // Working memory for FFTW operations
    fftw_complex* fftw_input_;   // Input buffer for FFTW
    fftw_complex* fftw_output_;  // Output buffer for FFTW

    // Processing parameters
    bool is_indoor_;           // Indoor/outdoor processing flag
    size_t angle_range_;       // Angular range in degrees (default: 90)
    size_t angle_resolution_;  // Angular resolution in degrees (default: 1)
    size_t angle_bins_;        // Number of angular bins

    // Frame processing statistics
    mutable std::atomic<size_t> frame_count_{0};

    /**
     * Preprocess frame from raw data to complex format
     * Equivalent to Python _preprocess_frame_from_raw_data
     * @param raw_data Raw radar data from DCA1000
     * @return ArrayFire array with complex radar frame
     */
    af::array preprocessFrameFromRawData(const RawDataVector& raw_data);

    /**
     * MUSIC-2D processing (stage 1) up to Doppler FFT and fftshift producing
     * range-doppler map. This corresponds to Python
     * process_3D_radar_frame_music_2d steps:
     *   - range FFT (range_processing)
     *   - doppler FFT (doppler_processing)
     *   - fftshift det_matrix (axis=1) and aoa_input (axis=2)
     * Remaining CFAR, detection, steering / MUSIC peak search are added in
     * later stages.
     *
     * Required inputs at this stage: complex frame + az/el search ranges and
     * algorithm knobs. (Knobs unused in stage 1 but accepted for interface
     * stability.)
     *
     * Returns AnalysisResult with ArrayFire range_doppler populated; other
     * heatmaps left empty (None) for later stages.
     */
    AnalysisResult processFrameMusic2DStage1(
        const af::array& complex_frame, int az_min = -53, int az_max = 53,
        int el_min = -18, int el_max = 18, int fine_az_step = 2,
        int fine_el_step = 4, int doppler_halfspan = 2, int coarse_az_step = 8,
        int coarse_el_step = 12, int fine_half_win_az = 8,
        int fine_half_win_el = 8, float music_diag_load = 0.01f,
        bool compute_tesseract = false);

    /**
     * Convert ArrayFire array to std::vector for output
     * @param af_array Input ArrayFire array
     * @return 2D vector representation
     */
    std::vector<std::vector<double>> arrayFireToVector2D(
        const af::array& af_array);

    /**
     * Initialize FFTW plans for the given dimensions
     * @param samples Number of range samples
     * @param chirps Number of chirps (for Doppler FFT)
     */
    void initializeFFTWPlans(size_t samples, size_t chirps);

    /**
     * Clean up FFTW resources
     */
    void cleanupFFTW();

    /**
     * Log processing performance statistics
     * @param processing_time Processing time in milliseconds
     */
    void logPerformanceStats(double processing_time) const;

    /**
     * Create empty/default analysis result for stub implementation
     * @param frame_timestamp Original frame timestamp
     * @param frame_number Frame sequence number
     * @param processing_time Processing time in milliseconds
     * @return Default analysis result with correct structure
     */
    AnalysisResult createDefaultResult(double frame_timestamp,
                                       size_t frame_number,
                                       double processing_time) const;
};

}  // namespace radar
