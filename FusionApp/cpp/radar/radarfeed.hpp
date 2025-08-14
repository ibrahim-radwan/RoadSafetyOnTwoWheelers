#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include "radardata.hpp"
#include "syncstate.hpp"
#include "threadsafequeue.hpp"

namespace radar
{

/**
 * Base interface for radar data feeds
 * Defines the contract for both live radar feeds and recording playback
 */
class RadarFeed
{
  public:
    virtual ~RadarFeed() = default;

    /**
     * Main execution method for the radar feed
     * This method should be called in a separate thread/process
     *
     * @param stream_queue Queue for sending radar frames to processing pipeline
     * @param stop_event Event to signal stopping the feed
     * @param control_queue Optional queue for receiving control commands
     * @param status_queue Optional queue for sending status updates
     */
    virtual void run(ThreadSafeQueue<std::shared_ptr<RadarFrame>>& stream_queue,
                     std::atomic<bool>& stop_event,
                     ThreadSafeQueue<std::string>* control_queue = nullptr,
                     ThreadSafeQueue<std::string>* status_queue = nullptr) = 0;

    /**
     * Initialize the radar feed
     * Called before run() to set up necessary resources
     *
     * @return true if initialization successful, false otherwise
     */
    virtual bool initialize() = 0;

    /**
     * Cleanup resources
     * Called after run() completes or when stopping
     */
    virtual void cleanup() = 0;

    /**
     * Check if the feed is ready to start
     * @return true if ready, false otherwise
     */
    virtual bool isReady() const = 0;

    /**
     * Get the current status of the feed
     * @return Status string describing current state
     */
    virtual std::string getStatus() const = 0;

  protected:
    // Common protected members for derived classes
    std::atomic<bool> _is_running{false};
    std::atomic<bool> _is_initialized{false};

    // Threading support
    std::unique_ptr<std::thread> _worker_thread;

    // Synchronization support
    std::shared_ptr<SyncState> _sync_state;
};

/**
 * Configuration structure for radar feeds
 */
struct RadarFeedConfig
{
    std::string config_file_path;
    std::string dest_dir;
    bool enable_sync = false;
    double timeout_seconds = 30.0;

    std::string toString() const
    {
        return "RadarFeedConfig{config_file=" + config_file_path +
               ", dest_dir=" + dest_dir +
               ", enable_sync=" + (enable_sync ? "true" : "false") +
               ", timeout=" + std::to_string(timeout_seconds) + "}";
    }
};

}  // namespace radar
