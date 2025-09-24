#include <af/array.h>
#include <af/image.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include "radar/config.hpp"
#include "radar/radardata.hpp"
#include "radar/radarheatmapanalyser.hpp"
#include "radar/recordingfeed.hpp"
#include "radar/threadsafequeue.hpp"

int main(int argc, char* argv[])
{
    std::string config_path;
    std::string dest_dir;

    // Parse command line arguments
    for (int i = 1; i < argc - 1; ++i)
    {
        if (std::string(argv[i]) == "--config-file")
        {
            config_path = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--dest-dir")
        {
            dest_dir = std::string(argv[i + 1]);
        }
    }

    if (config_path.empty() || dest_dir.empty())
    {
        std::cerr << "Usage: " << argv[0]
                  << " --config-file <path> [--dest-dir <directory>]"
                  << std::endl;
        std::cerr << "  --config-file: Path to radar configuration file"
                  << std::endl;
        std::cerr << "  --dest-dir: Directory containing recorded .bin files "
                     "(for playback mode)"
                  << std::endl;
        return 1;
    }

    try
    {
        radar::RadarConfig radar_config(config_path);
        std::cout << "Radar Configuration:" << std::endl;
        std::cout << radar_config.toString() << std::endl;

        // If dest_dir is provided, run in playback mode
        if (!dest_dir.empty())
        {
            std::cout << "\nStarting DCA1000 Recording Playback..."
                      << std::endl;
            std::cout << "Reading from directory: " << dest_dir << std::endl;

            // Create recording feed
            auto recording_feed = std::make_unique<radar::DCA1000Recording>(
                config_path, dest_dir);

            // Initialize the feed
            if (!recording_feed->initialize())
            {
                std::cerr << "Failed to initialize recording feed" << std::endl;
                return 1;
            }

            std::cout << "Recording feed initialized successfully" << std::endl;
            std::cout << "Total frames: " << recording_feed->getTotalFrames()
                      << std::endl;
            std::cout << "Total duration: "
                      << recording_feed->getTotalDuration() << " seconds"
                      << std::endl;
            std::cout << "Frame rate: " << recording_feed->getFrameRate()
                      << " Hz" << std::endl;

            // Create thread-safe queues for communication
            radar::ThreadSafeQueue<std::shared_ptr<radar::RadarFrame>>
                stream_queue;  // radar_input_queue
            radar::ThreadSafeQueue<std::string> control_queue;
            radar::ThreadSafeQueue<std::string> status_queue;
            radar::ThreadSafeQueue<radar::AnalysisResult>
                analysis_output_queue;  // radar_output_queue

            // Create stop event
            std::atomic<bool> stop_event{false};

            // Start playback in a separate thread
            std::thread playback_thread(
                [&]()
                {
                    try
                    {
                        recording_feed->run(stream_queue, stop_event,
                                            &control_queue, &status_queue);
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "Playback error: " << e.what()
                                  << std::endl;
                    }
                });

            // Initialize analyser
            radar::RadarHeatmapAnalyser analyser;
            if (!analyser.initialize(config_path))
            {
                std::cerr << "Failed to initialize RadarHeatmapAnalyser"
                          << std::endl;
                stop_event.store(true);
                playback_thread.join();
                return 1;
            }

            // Start analyser thread: consumes stream_queue, produces
            // analysis_output_queue
            std::thread analyser_thread(
                [&]()
                {
                    try
                    {
                        analyser.run(stream_queue, analysis_output_queue,
                                     stop_event);
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "Analyser error: " << e.what()
                                  << std::endl;
                    }
                });

            // Start automatic playback after a short delay
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            control_queue.push("play");

            // Consume analyser outputs for a few seconds as demonstration
            std::cout
                << "\nStarting analysis monitoring (will run for 10 seconds)..."
                << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            int result_count = 0;

            while (true)
            {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);

                if (elapsed.count() >= 10)
                {
                    break;  // Stop after 10 seconds
                }

                // Try to get an analysis result
                radar::AnalysisResult result;
                if (analysis_output_queue.waitAndPop(
                        result, std::chrono::milliseconds(100)))
                {
                    result_count++;
                    std::cout << "Result #" << result_count
                              << ": frame=" << result.frame_number
                              << ", time(ms)=" << result.processing_time_ms
                              << ", bins(r/d/a)=" << result.range_bins << "/"
                              << result.doppler_bins << "/"
                              << result.azimuth_bins
                              << ", points=" << result.point_cloud.size()
                              << std::endl;

                    // Save Range-Doppler heatmap as PNG for all frames
                    if (result.range_doppler.elements() > 0)
                    {
                        try
                        {
                            af::array rd = result.range_doppler.as(f32);
                            float min_val = af::min<float>(rd);
                            float max_val = af::max<float>(rd);

                            // Normalize to [0,1] range and transpose for proper
                            // orientation
                            af::array norm =
                                (rd - min_val) / (max_val - min_val + 1e-6f);
                            af::array img =
                                af::transpose(norm);  // (doppler, range)

                            // Convert to 8-bit for PNG
                            af::array img_u8 = (img * 255.0f).as(u8);
                            // Ensure output directory exists
                            std::filesystem::path out_dir =
                                std::filesystem::path(dest_dir) / "rd_png";
                            std::filesystem::create_directories(out_dir);
                            // Build filename matching radar frame basename:
                            // TTTTTTTTTT_TTTTT_FFFFFFFFFFFF_rd.png Reconstruct
                            // timestamp components from double with rounding
                            // guard
                            double ts = result.frame_timestamp;
                            long long ts_sec =
                                static_cast<long long>(std::floor(ts));
                            int ts_frac = static_cast<int>(std::round(
                                (ts - static_cast<double>(ts_sec)) * 1e5));
                            if (ts_frac >= 100000)
                            {
                                ts_sec += 1;
                                ts_frac = 0;
                            }
                            std::ostringstream base;
                            base << std::setw(10) << std::setfill('0') << ts_sec
                                 << "_" << std::setw(5) << std::setfill('0')
                                 << ts_frac << "_" << std::setw(12)
                                 << std::setfill('0') << result.frame_number;
                            std::string png_name = base.str() + "_rd.png";
                            auto out_path = (out_dir / png_name).string();
                            af::saveImageNative(out_path.c_str(), img_u8);
                            std::cout << "  Saved RD PNG: " << out_path
                                      << std::endl;
                        }
                        catch (const std::exception& e)
                        {
                            std::cerr << "  Failed to save RD PNG: " << e.what()
                                      << std::endl;
                        }
                    }
                }

                // Drain any status updates from the feed
                std::string status;
                if (status_queue.tryPop(status))
                {
                    std::cout << "Status: " << status << std::endl;
                }
            }

            std::cout << "\nTotal analysis results: " << result_count
                      << std::endl;

            // Stop playback and threads
            std::cout << "Stopping playback..." << std::endl;
            stop_event.store(true);
            analyser_thread.join();
            playback_thread.join();

            std::cout << "Playback completed successfully!" << std::endl;
        }
        else
        {
            std::cout << "\nNo destination directory specified. Radar "
                         "configuration loaded successfully."
                      << std::endl;
            std::cout << "Use --dest-dir <directory> to run in playback mode."
                      << std::endl;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}