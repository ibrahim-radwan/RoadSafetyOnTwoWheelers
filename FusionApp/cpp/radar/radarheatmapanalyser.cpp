#include "radarheatmapanalyser.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include "config.hpp"
#include "dsp/doppler_processing.hpp"
#include "dsp/range_processing.hpp"
#include "dsp/utils.hpp"
#include "exceptions.hpp"

namespace radar
{

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
    if (!config_file.empty())
    {
        initialize(config_file);
    }
}

RadarHeatmapAnalyser::~RadarHeatmapAnalyser()
{
    cleanupFFTW();
}

bool RadarHeatmapAnalyser::initialize(const std::string& config_file)
{
    try
    {
        config_file_path_ = config_file;

        // Load radar configuration
        // Directly construct AdcParams from config file; includes derived
        // fields
        adc_params_ = std::make_shared<AdcParams>(config_file);

        // Initialize ArrayFire (use CPU backend by default, can be changed to
        // GPU)
        af::setBackend(AF_BACKEND_CPU);
        af::info();  // Print ArrayFire info

        // Initialize FFTW plans
        initializeFFTWPlans(adc_params_->samples, adc_params_->chirps);

        is_initialized_ = true;

        std::cout << "RadarHeatmapAnalyser initialized successfully:"
                  << std::endl;
        std::cout << "  Config file: " << config_file << std::endl;
        std::cout << "  TX antennas: " << adc_params_->tx << std::endl;
        std::cout << "  RX antennas: " << adc_params_->rx << std::endl;
        std::cout << "  Samples: " << adc_params_->samples << std::endl;
        std::cout << "  Chirps: " << adc_params_->chirps << std::endl;
        std::cout << "  Range resolution: " << std::fixed
                  << std::setprecision(4) << adc_params_->range_resolution
                  << " m" << std::endl;
        std::cout << "  Doppler resolution: " << std::fixed
                  << std::setprecision(4) << adc_params_->doppler_resolution
                  << " m/s" << std::endl;
        // Derived resolutions are not present in AdcParams; skip printing them
        // for now

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to initialize RadarHeatmapAnalyser: " << e.what()
                  << std::endl;
        is_initialized_ = false;
        return false;
    }
}

AnalysisResult RadarHeatmapAnalyser::analyseFrame(
    const std::shared_ptr<RadarFrame>& frame)
{
    if (!is_initialized_)
    {
        throw RadarException("RadarHeatmapAnalyser not initialized");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Preprocess frame to complex format
    af::array complex_frame = preprocessFrameFromRawData(frame->getRawData());

    // Stage-1 MUSIC 2D processing (range & doppler FFT only)
    AnalysisResult result = processFrameMusic2DStage1(complex_frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0;

    // Set frame metadata
    result.frame_timestamp = frame->getTimestamp();
    result.frame_number = frame->getFrameNumber();

    // Log performance stats periodically
    if (frame_count_.load() % 10 == 0)
    {
        logPerformanceStats(result.processing_time_ms);
    }
    frame_count_++;

    return result;
}

void RadarHeatmapAnalyser::run(
    ThreadSafeQueue<std::shared_ptr<RadarFrame>>& input_queue,
    ThreadSafeQueue<AnalysisResult>& output_queue, std::atomic<bool>& stop_flag)
{
    if (!is_initialized_)
    {
        std::cerr
            << "RadarHeatmapAnalyser not initialized, cannot start processing"
            << std::endl;
        return;
    }

    std::cout << "RadarHeatmapAnalyser processing thread started" << std::endl;

    while (!stop_flag.load())
    {
        try
        {
            std::shared_ptr<RadarFrame> frame_ptr;
            if (input_queue.waitAndPop(frame_ptr,
                                       std::chrono::milliseconds(100)))
            {
                std::cout << "[RadarHeatmapAnalyser] Received frame "
                          << frame_ptr->getFrameNumber() << std::endl;
                AnalysisResult result = analyseFrame(frame_ptr);
                output_queue.push(std::move(result));
            }
        }
        catch (const std::exception& e)
        {
            std::cerr
                << "Error in RadarHeatmapAnalyser processing (shared_ptr): "
                << e.what() << std::endl;
        }
    }

    std::cout << "RadarHeatmapAnalyser processing thread stopped" << std::endl;
}

af::array RadarHeatmapAnalyser::preprocessFrameFromRawData(
    const RawDataVector& raw_data)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    // Implements Python _preprocess_frame_from_raw_data logic:
    // Python sequence:
    //   frame = reshape(raw, (chirps, tx, samples, IQ, rx))
    //   frame = transpose(frame, (0, 1, 4, 2, 3))  # -> (chirps, tx, rx,
    //   samples, IQ) complex_frame = (1j * frame[...,1] + frame[...,0])  # I
    //   first return complex_frame with shape (chirps, tx, rx, samples)
    // Raw data layout (fastest to slowest index in the 1D vector):
    //   r (rx) -> iq (IQ) -> s (samples) -> t (tx) -> c (chirps)

    if (!adc_params_)
    {
        throw RadarException("ADC parameters not available for preprocessing");
    }

    const size_t chirps = static_cast<size_t>(adc_params_->chirps);
    const size_t tx = static_cast<size_t>(adc_params_->tx);
    const size_t samples = static_cast<size_t>(adc_params_->samples);
    const size_t iq = static_cast<size_t>(adc_params_->IQ);  // must be 2 (I,Q)
    const size_t rx = static_cast<size_t>(adc_params_->rx);

    if (iq != 2)
    {
        throw RadarException("IQ dimension must be 2");
    }

    const size_t expected_size = chirps * tx * samples * iq * rx;
    if (raw_data.size() != expected_size)
    {
        std::ostringstream oss;
        oss << "Raw data size mismatch: got " << raw_data.size()
            << ", expected " << expected_size << " (chirps=" << chirps
            << ", tx=" << tx << ", samples=" << samples << ", iq=" << iq
            << ", rx=" << rx << ")";
        throw RadarException(oss.str());
    }

    // Python index calculation: chirp*tx*samples*IQ*rx + tx*samples*IQ*rx +
    // sample*IQ*rx + IQ*rx + rx
    auto py_idx = [&](int c, int t, int s, int iq, int r) -> size_t
    {
        return c * (tx * samples * 2 * rx) + t * (samples * 2 * rx) +
               s * (2 * rx) + iq * rx + r;
    };

    // Step 1: Create arrays manually to match Python's exact indexing
    // We need to extract I and Q values using the same indexing as Python

    // Pre-allocate I and Q arrays with final target shape: (chirps, tx, rx,
    // samples)
    std::vector<int16_t> I_data(chirps * tx * rx * samples);
    std::vector<int16_t> Q_data(chirps * tx * rx * samples);

    // Extract I and Q values using Python's exact indexing logic
    for (int c = 0; c < chirps; c++)
    {
        for (int t = 0; t < tx; t++)
        {
            for (int r = 0; r < rx; r++)
            {
                for (int s = 0; s < samples; s++)
                {
                    // Python index: c*tx*samples*IQ*rx + t*samples*IQ*rx +
                    // s*IQ*rx + iq*rx + r
                    size_t py_i_idx =
                        py_idx(c, t, s, 0, r);  // I channel (iq=0)
                    size_t py_q_idx =
                        py_idx(c, t, s, 1, r);  // Q channel (iq=1)

                    // Target index in our I/Q arrays: c*tx*rx*samples +
                    // t*rx*samples + r*samples + s
                    size_t target_idx = c * (tx * rx * samples) +
                                        t * (rx * samples) + r * samples + s;

                    I_data[target_idx] = raw_data[py_i_idx];
                    Q_data[target_idx] = raw_data[py_q_idx];
                }
            }
        }
    }

    // Create ArrayFire arrays with correct shape: (samples, rx, tx, chirps)
    // Note: ArrayFire uses column-major ordering, so we reverse the dimensions
    af::array I_int16(samples, rx, tx, chirps, I_data.data());
    af::array Q_int16(samples, rx, tx, chirps, Q_data.data());

    // Convert to float32
    af::array I_f32 = I_int16.as(f32);
    af::array Q_f32 = Q_int16.as(f32);

    // Create complex array
    af::array complex_temp = af::complex(I_f32, Q_f32);

    // Reorder from (samples, rx, tx, chirps) to (chirps, tx, rx, samples) to
    // match Python
    af::array complex_iq = af::reorder(complex_temp, 3, 2, 1, 0);

    // Final assertion to ensure output shape matches Python expectation
    {
        af::dim4 d = complex_iq.dims();
        if (d[0] != (dim_t)chirps || d[1] != (dim_t)tx || d[2] != (dim_t)rx ||
            d[3] != (dim_t)samples)
        {
            std::ostringstream oss;
            oss << "Final complex frame shape mismatch: got (" << d[0] << ","
                << d[1] << "," << d[2] << "," << d[3] << "), expected ("
                << chirps << "," << tx << "," << rx << "," << samples << ")";
            throw RadarException(oss.str());
        }
    }

    // Ensure complex32 type
    if (complex_iq.type() != c32)
    {
        complex_iq = complex_iq.as(c32);
    }

    auto preproc_end = std::chrono::high_resolution_clock::now();
    auto preproc_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          preproc_end - t_start)
                          .count();
    std::cout << "[Profile] Preprocessing: " << preproc_us << " us"
              << std::endl;

    return complex_iq;
}

AnalysisResult RadarHeatmapAnalyser::processFrameMusic2DStage1(
    const af::array& complex_frame, int az_min, int az_max, int el_min,
    int el_max, int fine_az_step, int fine_el_step, int doppler_halfspan,
    int coarse_az_step, int coarse_el_step, int fine_half_win_az,
    int fine_half_win_el, float music_diag_load, bool compute_tesseract)
{
    (void)az_min;
    (void)az_max;
    (void)el_min;
    (void)el_max;
    (void)fine_az_step;
    (void)fine_el_step;
    (void)doppler_halfspan;
    (void)coarse_az_step;
    (void)coarse_el_step;
    (void)fine_half_win_az;
    (void)fine_half_win_el;
    (void)music_diag_load;
    (void)compute_tesseract;  // Unused in stage 1

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!adc_params_)
        throw RadarException("ADC params not set");
    // Expect complex_frame dims: (chirps, tx, rx, samples)
    af::dim4 d = complex_frame.dims();
    if ((size_t)d[0] != (size_t)adc_params_->chirps ||
        (size_t)d[1] != (size_t)adc_params_->tx ||
        (size_t)d[2] != (size_t)adc_params_->rx ||
        (size_t)d[3] != (size_t)adc_params_->samples)
    {
        std::ostringstream oss;
        oss << "processFrameMusic2DStage1: unexpected frame shape (" << d[0]
            << "," << d[1] << "," << d[2] << "," << d[3] << ") expected ("
            << adc_params_->chirps << "," << adc_params_->tx << ","
            << adc_params_->rx << "," << adc_params_->samples << ")";
        throw RadarException(oss.str());
    }

    // 1) Range processing (window + FFT over samples) producing radar cube
    // (chirps, vrx, samples)
    bool disable_window = false;  // Use windowing to match Python
    auto win_sel = disable_window ? dsp::Window::NONE : dsp::Window::HAMMING;

    auto range_start = std::chrono::high_resolution_clock::now();
    af::array radar_cube = dsp::range_processing(
        complex_frame, adc_params_->chirps, adc_params_->tx, adc_params_->rx,
        adc_params_->samples, win_sel);
    auto range_end = std::chrono::high_resolution_clock::now();
    auto range_us = std::chrono::duration_cast<std::chrono::microseconds>(
                        range_end - range_start)
                        .count();
    std::cout << "[Profile] Range processing: " << range_us << " us"
              << std::endl;

    // 2) Doppler processing (window + FFT over chirps)
    auto doppler_start = std::chrono::high_resolution_clock::now();
    af::array doppler_fft_result = dsp::doppler_processing(
        radar_cube, adc_params_->chirps, adc_params_->tx * adc_params_->rx,
        adc_params_->samples, win_sel);
    auto doppler_end = std::chrono::high_resolution_clock::now();
    auto doppler_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          doppler_end - doppler_start)
                          .count();
    std::cout << "[Profile] Doppler processing: " << doppler_us << " us"
              << std::endl;

    // Convert doppler_fft_result -> det_matrix (range,doppler) like Python:
    // log(abs(FFT)) accumulation over vrx.
    auto make_range_doppler = [&](const af::array& doppler_cube) -> af::array
    {
        // doppler_cube: (chirps, vrx, samples)
        // Reorder to (range, vrx, doppler) to mirror Python interleaved=False
        af::array reordered =
            af::reorder(doppler_cube, 2, 1, 0);  // (range, vrx, doppler)
        // Magnitude then log2 like python (fft2d_log_abs = log2(|fft2d_out|))
        af::array mag = af::abs(reordered);
        // Avoid log(0): add tiny epsilon
        af::array log_mag = af::log2(mag + 1e-12f);
        // Accumulate over vrx (axis=1) -> (range, 1, doppler)
        af::array det = af::sum(log_mag, 1);
        // Squeeze to exactly 2D (range, doppler) with singleton dimensions = 1
        det = af::moddims(det, det.dims(0), det.dims(2), 1, 1);
        return det;
    };

    auto rd_start = std::chrono::high_resolution_clock::now();
    af::array det_matrix = make_range_doppler(doppler_fft_result);

    // fftshift along Doppler axis for RD (axis=1 when det_matrix is (range,
    // doppler))
    af::array det_matrix_shifted = dsp::fftshift_dim(det_matrix, 1);
    auto rd_end = std::chrono::high_resolution_clock::now();
    auto rd_us =
        std::chrono::duration_cast<std::chrono::microseconds>(rd_end - rd_start)
            .count();
    std::cout << "[Profile] Range-Doppler matrix: " << rd_us << " us"
              << std::endl;

    // For future AoA input parity with Python, also compute a shifted Doppler
    // cube by first reordering to (range, vrx, doppler) then shifting along
    // doppler (axis=2).
    af::array doppler_fft_shifted =
        dsp::fftshift_dim(af::reorder(doppler_fft_result, 2, 1, 0), 2);
    (void)doppler_fft_shifted;  // reserved for future aoa_input

    // Populate AnalysisResult
    AnalysisResult result = createDefaultResult(0.0, 0, 0.0);
    result.range_bins = adc_params_->samples;
    result.doppler_bins = adc_params_->chirps;
    result.azimuth_bins = angle_bins_;

    // Store as ArrayFire array only
    result.range_doppler = det_matrix_shifted;  // (range, doppler)

    // range_azimuth left empty for stage 1

    auto t1 = std::chrono::high_resolution_clock::now();
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    result.processing_time_ms = us / 1000.0;
    return result;
}

std::vector<std::vector<double>> RadarHeatmapAnalyser::arrayFireToVector2D(
    const af::array& af_array)
{
    // Convert ArrayFire array to 2D vector
    if (af_array.numdims() != 2)
    {
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
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result[i][j] = host_data[i * cols + j];
        }
    }

    return result;
}

void RadarHeatmapAnalyser::initializeFFTWPlans(size_t samples, size_t chirps)
{
    // Allocate FFTW memory
    fftw_input_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samples);
    fftw_output_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * samples);

    if (!fftw_input_ || !fftw_output_)
    {
        throw RadarException("Failed to allocate FFTW memory");
    }

    // Create FFTW plans
    range_fft_plan_ = fftw_plan_dft_1d(samples, fftw_input_, fftw_output_,
                                       FFTW_FORWARD, FFTW_MEASURE);

    // Doppler FFT plan (for future use)
    doppler_fft_plan_ = fftw_plan_dft_1d(chirps, fftw_input_, fftw_output_,
                                         FFTW_FORWARD, FFTW_MEASURE);

    if (!range_fft_plan_ || !doppler_fft_plan_)
    {
        throw RadarException("Failed to create FFTW plans");
    }

    std::cout << "FFTW plans initialized for " << samples
              << " range samples and " << chirps << " chirps" << std::endl;
}

void RadarHeatmapAnalyser::cleanupFFTW()
{
    if (range_fft_plan_)
    {
        fftw_destroy_plan(range_fft_plan_);
        range_fft_plan_ = nullptr;
    }

    if (doppler_fft_plan_)
    {
        fftw_destroy_plan(doppler_fft_plan_);
        doppler_fft_plan_ = nullptr;
    }

    if (fftw_input_)
    {
        fftw_free(fftw_input_);
        fftw_input_ = nullptr;
    }

    if (fftw_output_)
    {
        fftw_free(fftw_output_);
        fftw_output_ = nullptr;
    }

    fftw_cleanup();
}

void RadarHeatmapAnalyser::logPerformanceStats(double processing_time) const
{
    std::cout << "[RadarHeatmapAnalyser] Frame " << frame_count_.load()
              << " processed in " << std::fixed << std::setprecision(2)
              << processing_time << " ms" << std::endl;
}

AnalysisResult RadarHeatmapAnalyser::createDefaultResult(
    double frame_timestamp, size_t frame_number, double processing_time) const
{
    AnalysisResult result;

    result.frame_timestamp = frame_timestamp;
    result.frame_number = frame_number;
    result.processing_time_ms = processing_time;

    // Set dimensions
    if (adc_params_)
    {
        result.range_bins = adc_params_->samples;
        result.doppler_bins = adc_params_->chirps;
    }
    result.azimuth_bins = angle_bins_;

    return result;
}

std::string RadarHeatmapAnalyser::toString() const
{
    std::ostringstream oss;
    oss << "RadarHeatmapAnalyser{"
        << "initialized=" << is_initialized_.load()
        << ", config_file=" << config_file_path_ << ", is_indoor=" << is_indoor_
        << ", angle_range=" << angle_range_
        << ", angle_resolution=" << angle_resolution_
        << ", frames_processed=" << frame_count_.load();

    if (adc_params_)
    {
        oss << ", tx=" << adc_params_->tx << ", rx=" << adc_params_->rx
            << ", samples=" << adc_params_->samples
            << ", chirps=" << adc_params_->chirps;
    }

    oss << "}";
    return oss.str();
}

}  // namespace radar
