#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include "radar/config.hpp"
#include "radar/radardata.hpp"
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

    if (config_path.empty())
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
                stream_queue;
            radar::ThreadSafeQueue<std::string> control_queue;
            radar::ThreadSafeQueue<std::string> status_queue;

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

            // Start automatic playback after a short delay
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            control_queue.push("play");

            // Monitor frames for a few seconds as demonstration
            std::cout
                << "\nStarting frame monitoring (will run for 10 seconds)..."
                << std::endl;
            auto start_time = std::chrono::steady_clock::now();
            int frame_count = 0;

            while (true)
            {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time);

                if (elapsed.count() >= 10)
                {
                    break;  // Stop after 10 seconds
                }

                // Try to get a frame
                std::shared_ptr<radar::RadarFrame> frame;
                if (stream_queue.tryPop(frame))
                {
                    frame_count++;
                    if (frame_count % 10 == 0)
                    {
                        std::cout << "Received frame " << frame_count
                                  << " (timestamp: " << frame->getTimestamp()
                                  << "s)" << std::endl;
                    }
                }

                // Check for status updates
                std::string status;
                if (status_queue.tryPop(status))
                {
                    std::cout << "Status: " << status << std::endl;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            std::cout << "\nTotal frames processed: " << frame_count
                      << std::endl;

            // Stop playback
            std::cout << "Stopping playback..." << std::endl;
            stop_event.store(true);
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