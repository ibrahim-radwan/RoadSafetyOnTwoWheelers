#pragma once

#include <string>
#include <vector>
#include "config.hpp"
#include "radarfeed.hpp"

namespace radar
{

/**
 * Frame file information structure
 * Contains filepath, timestamp, and frame number for each recorded frame
 */
struct FrameFileInfo
{
    std::string filepath;
    double timestamp;
    size_t frame_number;

    FrameFileInfo(const std::string& path, double ts, size_t frame_num)
        : filepath(path), timestamp(ts), frame_number(frame_num)
    {
    }
};

/**
 * DCA1000 Recording playback class that reads recorded .bin files from a
 * directory and plays them back with timing control, navigation, and
 * synchronized playback
 */
class DCA1000Recording : public RadarFeed
{
  private:
    // Configuration
    std::string _dest_dir;
    std::shared_ptr<AdcParams> _adc_params;

    // Frame data
    std::vector<FrameFileInfo> _frame_files;
    std::atomic<size_t> _current_frame_index{0};
    double _frame_rate = 10.0;  // Default frame rate

    // Playback state
    std::atomic<PlaybackState> _playback_state{PlaybackState::STOPPED};

    // Threading
    std::unique_ptr<std::thread> _sender_thread;
    std::atomic<bool> _stop_requested{false};

    // Status updates
    ThreadSafeQueue<std::string>* _status_queue = nullptr;

  public:
    /**
     * Constructor
     * @param config_file_path Path to radar configuration file
     * @param dest_dir Directory containing recorded .bin files
     * @param sync_state Optional shared synchronization state for multi-feed
     * coordination
     */
    explicit DCA1000Recording(const std::string& config_file_path,
                              const std::string& dest_dir,
                              std::shared_ptr<SyncState> sync_state = nullptr);

    virtual ~DCA1000Recording();

    // RadarFeed interface implementation
    void run(ThreadSafeQueue<std::shared_ptr<RadarFrame>>& stream_queue,
             std::atomic<bool>& stop_event,
             ThreadSafeQueue<std::string>* control_queue = nullptr,
             ThreadSafeQueue<std::string>* status_queue = nullptr) override;

    bool initialize() override;
    void cleanup() override;
    bool isReady() const override;
    std::string getStatus() const override;

    // Playback control methods
    void play();
    void pause();
    void stop();
    void seekToFrame(size_t frame_index);
    void seekToTime(double timestamp);
    void seekToPercent(double percent);

    // Information getters
    size_t getCurrentFrameIndex() const
    {
        return _current_frame_index.load();
    }
    size_t getTotalFrames() const
    {
        return _frame_files.size();
    }
    double getCurrentTimestamp() const;
    double getTotalDuration() const;
    double getFrameRate() const
    {
        return _frame_rate;
    }

  private:
    // Initialization methods
    void scanRecordingFiles();
    void loadRadarConfig(const std::string& config_file_path);
    void validateConfiguration();

    // File operations
    std::shared_ptr<RadarFrame> readFrameFromFile(
        const FrameFileInfo& frame_info);
    bool parseFilename(const std::string& filename, double& timestamp,
                       size_t& frame_number);

    // Threading methods
    void senderThreadFunction(
        ThreadSafeQueue<std::shared_ptr<RadarFrame>>& stream_queue,
        std::atomic<bool>& stop_event,
        ThreadSafeQueue<std::string>* control_queue);

    // Synchronization methods
    bool shouldSendFrame(const FrameFileInfo& frame_info);
    void waitForFrameTime(const FrameFileInfo& frame_info,
                          std::atomic<bool>& stop_event);
    void handleSeekOperation();

    // Control command processing
    void handleControlCommand(const std::string& command);
    void sendStatusUpdate();

    // Validation helpers
    void validateFrameFile(const std::string& filepath);
    void validateDestDirectory();
};

}  // namespace radar
