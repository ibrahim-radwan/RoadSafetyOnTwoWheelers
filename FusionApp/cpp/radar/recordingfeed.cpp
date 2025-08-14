#include "recordingfeed.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <thread>
#include "exceptions.hpp"

namespace radar
{

DCA1000Recording::DCA1000Recording(const std::string& config_file_path,
                                   const std::string& dest_dir,
                                   std::shared_ptr<SyncState> sync_state)
    : _dest_dir(dest_dir)
{
    _sync_state = sync_state;

    try
    {
        loadRadarConfig(config_file_path);
    }
    catch (const std::exception& e)
    {
        throw ConfigException("Failed to load radar configuration: " +
                              std::string(e.what()));
    }
}

DCA1000Recording::~DCA1000Recording()
{
    cleanup();
}

bool DCA1000Recording::initialize()
{
    if (_is_initialized.load())
    {
        return true;
    }

    try
    {
        std::cout << "Validating destination directory..." << std::endl;
        validateDestDirectory();

        std::cout << "Scanning recording files..." << std::endl;
        scanRecordingFiles();
        std::cout << "Found " << _frame_files.size() << " frame files"
                  << std::endl;

        std::cout << "Validating configuration..." << std::endl;
        validateConfiguration();

        _is_initialized.store(true);

        // Signal readiness for synchronized mode
        if (_sync_state)
        {
            SyncStateUtils::signalFeedReady(*_sync_state);
        }

        std::cout << "DCA1000Recording initialized successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cout << "DCA1000Recording initialization failed: " << e.what()
                  << std::endl;
        _is_initialized.store(false);
        return false;
    }
}

void DCA1000Recording::cleanup()
{
    _stop_requested.store(true);

    if (_sender_thread && _sender_thread->joinable())
    {
        _sender_thread->join();
    }

    _is_running.store(false);
    _is_initialized.store(false);
}

bool DCA1000Recording::isReady() const
{
    return _is_initialized.load() && !_frame_files.empty() &&
           _adc_params != nullptr;
}

std::string DCA1000Recording::getStatus() const
{
    std::ostringstream oss;
    oss << "DCA1000Recording{";
    oss << "state=" << static_cast<int>(_playback_state.load()) << ", ";
    oss << "frame=" << _current_frame_index.load() << "/" << _frame_files.size()
        << ", ";
    oss << "ready=" << (isReady() ? "true" : "false") << ", ";
    oss << "running=" << (_is_running.load() ? "true" : "false");
    oss << "}";
    return oss.str();
}

void DCA1000Recording::run(
    ThreadSafeQueue<std::shared_ptr<RadarFrame>>& stream_queue,
    std::atomic<bool>& stop_event, ThreadSafeQueue<std::string>* control_queue,
    ThreadSafeQueue<std::string>* status_queue)
{
    if (!initialize())
    {
        throw std::runtime_error("Failed to initialize DCA1000Recording");
    }

    _status_queue = status_queue;
    _is_running.store(true);

    // Send ADC parameters first (similar to Python implementation)
    if (!_adc_params)
    {
        throw std::runtime_error("ADC parameters not loaded");
    }

    // Send initial ADC parameters as a special frame (like Python version)
    // For now, we'll skip this and focus on frame sending

    // Create and start the frame sender thread
    _sender_thread = std::make_unique<std::thread>(
        [this, &stream_queue, &stop_event, control_queue]()
        { senderThreadFunction(stream_queue, stop_event, control_queue); });

    // Main loop - handle control commands and wait for stop event
    while (!stop_event.load() && !_stop_requested.load())
    {
        // Check for control commands
        if (control_queue)
        {
            std::string command;
            if (control_queue->tryPop(command))
            {
                handleControlCommand(command);
            }
        }

        // Send periodic status updates
        sendStatusUpdate();

        // Sleep briefly to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Wait for sender thread to finish
    _stop_requested.store(true);
    if (_sender_thread && _sender_thread->joinable())
    {
        _sender_thread->join();
    }

    _is_running.store(false);
}

void DCA1000Recording::play()
{
    if (_current_frame_index.load() >= _frame_files.size())
    {
        return;  // At end of recording
    }

    if (_sync_state)
    {
        SyncStateUtils::setPlaybackState(*_sync_state, PlaybackState::PLAYING);
    }
    else
    {
        _playback_state.store(PlaybackState::PLAYING);
    }

    sendStatusUpdate();
}

void DCA1000Recording::pause()
{
    if (_sync_state)
    {
        SyncStateUtils::setPlaybackState(*_sync_state, PlaybackState::PAUSED);
    }
    else
    {
        _playback_state.store(PlaybackState::PAUSED);
    }

    sendStatusUpdate();
}

void DCA1000Recording::stop()
{
    if (_sync_state)
    {
        SyncStateUtils::setPlaybackState(*_sync_state, PlaybackState::STOPPED);
        SyncStateUtils::seekToTime(*_sync_state, 0.0);
    }
    else
    {
        _playback_state.store(PlaybackState::STOPPED);
    }

    _current_frame_index.store(0);
    sendStatusUpdate();
}

void DCA1000Recording::seekToFrame(size_t frame_index)
{
    if (frame_index >= _frame_files.size())
    {
        throw std::out_of_range("Frame index out of range");
    }

    _current_frame_index.store(frame_index);

    // Update sync state if available
    if (_sync_state && !_frame_files.empty())
    {
        double target_timestamp = _frame_files[frame_index].timestamp;
        double start_timestamp = _frame_files[0].timestamp;
        double relative_time = target_timestamp - start_timestamp;
        SyncStateUtils::seekToTime(*_sync_state, relative_time);
    }

    sendStatusUpdate();
}

void DCA1000Recording::seekToTime(double timestamp)
{
    // Find closest frame to target timestamp
    size_t best_index = 0;
    double best_diff = std::numeric_limits<double>::max();

    for (size_t i = 0; i < _frame_files.size(); ++i)
    {
        double diff = std::abs(_frame_files[i].timestamp - timestamp);
        if (diff < best_diff)
        {
            best_diff = diff;
            best_index = i;
        }
    }

    seekToFrame(best_index);
}

void DCA1000Recording::seekToPercent(double percent)
{
    if (percent < 0.0 || percent > 100.0)
    {
        throw std::invalid_argument("Percent must be between 0 and 100");
    }

    if (_frame_files.empty())
    {
        return;
    }

    size_t target_index =
        static_cast<size_t>((percent / 100.0) * (_frame_files.size() - 1));
    seekToFrame(target_index);
}

double DCA1000Recording::getCurrentTimestamp() const
{
    size_t current_index = _current_frame_index.load();
    if (current_index >= _frame_files.size())
    {
        return 0.0;
    }
    return _frame_files[current_index].timestamp;
}

double DCA1000Recording::getTotalDuration() const
{
    if (_frame_files.empty())
    {
        return 0.0;
    }
    return _frame_files.back().timestamp - _frame_files.front().timestamp;
}

// Private implementation methods

void DCA1000Recording::scanRecordingFiles()
{
    if (!std::filesystem::exists(_dest_dir))
    {
        throw DirectoryException("Recording directory does not exist: " +
                                 _dest_dir);
    }

    std::vector<FrameFileInfo> frame_info;
    std::regex filename_pattern(R"((\d{10})_(\d{5})_(\d{12})\.bin$)");

    std::cout << "Scanning directory: " << _dest_dir << std::endl;

    int total_files = 0;
    int bin_files = 0;
    int parsed_files = 0;

    for (const auto& entry : std::filesystem::directory_iterator(_dest_dir))
    {
        total_files++;

        if (!entry.is_regular_file())
        {
            std::cout << "  Skipping non-file: "
                      << entry.path().filename().string() << std::endl;
            continue;
        }

        if (entry.path().extension() != ".bin")
        {
            std::cout << "  Skipping non-bin file: "
                      << entry.path().filename().string() << std::endl;
            continue;
        }

        bin_files++;

        std::string filename = entry.path().filename().string();
        double timestamp;
        size_t frame_number;

        if (parseFilename(filename, timestamp, frame_number))
        {
            frame_info.emplace_back(entry.path().string(), timestamp,
                                    frame_number);
            parsed_files++;
            std::cout << "  Parsed: " << filename
                      << " -> frame=" << frame_number
                      << ", timestamp=" << timestamp << std::endl;
        }
        else
        {
            std::cout << "  Failed to parse filename: " << filename
                      << " (expected format: NNNNNNNNNN_NNNNN_NNNNNNNNNNNN.bin)"
                      << std::endl;
        }
    }

    std::cout << "File scan summary: " << total_files << " total, " << bin_files
              << " .bin files, " << parsed_files << " successfully parsed"
              << std::endl;

    if (frame_info.empty())
    {
        throw FileNotFoundException("No valid .bin files found in directory: " +
                                    _dest_dir);
    }

    // Sort by timestamp
    std::sort(frame_info.begin(), frame_info.end(),
              [](const FrameFileInfo& a, const FrameFileInfo& b)
              { return a.timestamp < b.timestamp; });

    _frame_files = std::move(frame_info);
}

void DCA1000Recording::loadRadarConfig(const std::string& config_file_path)
{
    // Use the existing RadarConfig class
    RadarConfig config(config_file_path);

    // Create shared AdcParams from config
    _adc_params = std::make_shared<AdcParams>(config.getAdcParams());

    // Extract frame rate - frame_periodicity is in milliseconds
    // Convert to Hz: 1000ms / frame_periodicity_ms = frames per second
    _frame_rate = 1000.0 / _adc_params->frame_periodicity;

    std::cout << "Calculated frame rate: " << _frame_rate
              << " Hz (from frame_periodicity="
              << _adc_params->frame_periodicity << " ms)" << std::endl;
}

void DCA1000Recording::validateConfiguration()
{
    if (!_adc_params)
    {
        throw ConfigException("ADC parameters not loaded");
    }

    if (_frame_files.empty())
    {
        throw ConfigException("No frame files found");
    }

    std::cout << "Validating " << _frame_files.size() << " frame files..."
              << std::endl;

    // Validate that frame files exist and are readable
    int valid_files = 0;
    for (const auto& frame_info : _frame_files)
    {
        try
        {
            validateFrameFile(frame_info.filepath);
            valid_files++;
        }
        catch (const std::exception& e)
        {
            std::cout << "  File validation failed for " << frame_info.filepath
                      << ": " << e.what() << std::endl;
            throw;  // Re-throw to fail initialization
        }
    }

    std::cout << "All " << valid_files << " frame files validated successfully"
              << std::endl;
}

std::shared_ptr<RadarFrame> DCA1000Recording::readFrameFromFile(
    const FrameFileInfo& frame_info)
{
    std::ifstream file(frame_info.filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw FileNotFoundException("Cannot open frame file: " +
                                    frame_info.filepath);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size == 0)
    {
        throw InvalidFrameException("Empty frame file: " + frame_info.filepath);
    }

    // Read raw data
    std::vector<int16_t> raw_data(file_size / sizeof(int16_t));
    file.read(reinterpret_cast<char*>(raw_data.data()), file_size);

    if (file.gcount() != static_cast<std::streamsize>(file_size))
    {
        throw InvalidFrameException("Failed to read complete frame file: " +
                                    frame_info.filepath);
    }

    // Create RadarFrame using factory
    return RadarFrameFactory::createFromFileData(
        frame_info.timestamp, raw_data, frame_info.frame_number, _adc_params);
}

bool DCA1000Recording::parseFilename(const std::string& filename,
                                     double& timestamp, size_t& frame_number)
{
    std::regex pattern(R"((\d{10})_(\d{5})_(\d{12})\.bin$)");
    std::smatch match;

    if (!std::regex_match(filename, match, pattern))
    {
        return false;
    }

    try
    {
        int timestamp_int = std::stoi(match[1].str());
        int timestamp_frac = std::stoi(match[2].str());
        frame_number = std::stoull(match[3].str());

        timestamp = timestamp_int + (timestamp_frac / 1e5);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void DCA1000Recording::senderThreadFunction(
    ThreadSafeQueue<std::shared_ptr<RadarFrame>>& stream_queue,
    std::atomic<bool>& stop_event, ThreadSafeQueue<std::string>* control_queue)
{
    bool use_sync = (_sync_state != nullptr);
    double last_timeline_position = 0.0;

    if (use_sync)
    {
        // Wait for start signal in synchronized mode
        if (!SyncStateUtils::waitForStartSignal(*_sync_state, 30.0))
        {
            return;  // Timeout
        }
    }

    while (!stop_event.load() && !_stop_requested.load())
    {
        try
        {
            // Check playback state
            PlaybackState current_state;
            if (use_sync)
            {
                current_state = SyncStateUtils::getPlaybackState(*_sync_state);
            }
            else
            {
                current_state = _playback_state.load();
            }

            if (current_state == PlaybackState::PLAYING)
            {
                size_t current_index = _current_frame_index.load();

                // Check if we have more frames
                if (current_index >= _frame_files.size())
                {
                    // End of recording reached
                    if (use_sync)
                    {
                        SyncStateUtils::setPlaybackState(
                            *_sync_state, PlaybackState::STOPPED);
                    }
                    else
                    {
                        _playback_state.store(PlaybackState::STOPPED);
                    }
                    continue;
                }

                const auto& frame_info = _frame_files[current_index];

                // Wait for correct timing
                waitForFrameTime(frame_info, stop_event);

                // Check if we should still send the frame
                if (stop_event.load() || _stop_requested.load())
                {
                    break;
                }

                PlaybackState check_state;
                if (use_sync)
                {
                    check_state =
                        SyncStateUtils::getPlaybackState(*_sync_state);
                }
                else
                {
                    check_state = _playback_state.load();
                }

                if (check_state != PlaybackState::PLAYING)
                {
                    continue;
                }

                // Read and send frame
                try
                {
                    auto frame = readFrameFromFile(frame_info);
                    stream_queue.push(frame);

                    // Update tracking
                    if (use_sync)
                    {
                        SyncStateUtils::setLastRadarTimestamp(
                            *_sync_state, frame_info.timestamp);
                    }

                    // Advance to next frame
                    _current_frame_index.store(current_index + 1);

                    // Send status update every few frames
                    if ((current_index % 5) == 0)
                    {
                        sendStatusUpdate();
                    }
                }
                catch (const std::exception& e)
                {
                    // Skip this frame and continue
                    _current_frame_index.store(current_index + 1);
                    continue;
                }
            }
            else
            {
                // Paused or stopped - just wait
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        catch (const std::exception& e)
        {
            // Log error and continue
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void DCA1000Recording::waitForFrameTime(const FrameFileInfo& frame_info,
                                        std::atomic<bool>& stop_event)
{
    if (!_sync_state)
    {
        // Legacy timing mode - use frame rate
        double frame_period = 1.0 / _frame_rate;
        std::this_thread::sleep_for(
            std::chrono::duration<double>(frame_period));
        return;
    }

    // Synchronized timing mode
    double start_timestamp = SyncStateUtils::getStartTimestamp(*_sync_state);
    double relative_frame_time = frame_info.timestamp - start_timestamp;

    while (!stop_event.load() && !_stop_requested.load())
    {
        double current_timeline =
            SyncStateUtils::getCurrentTimelinePosition(*_sync_state);

        if (current_timeline >= relative_frame_time)
        {
            break;  // Time to send this frame
        }

        // Check if playback was paused while waiting
        if (SyncStateUtils::getPlaybackState(*_sync_state) !=
            PlaybackState::PLAYING)
        {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void DCA1000Recording::handleControlCommand(const std::string& command)
{
    if (command == "play")
    {
        play();
    }
    else if (command == "pause")
    {
        pause();
    }
    else if (command == "stop")
    {
        stop();
    }
    else if (command.substr(0, 5) == "seek:")
    {
        try
        {
            size_t position = std::stoull(command.substr(5));
            seekToFrame(position);
        }
        catch (const std::exception&)
        {
            // Invalid seek command, ignore
        }
    }
}

void DCA1000Recording::sendStatusUpdate()
{
    if (!_status_queue)
    {
        return;
    }

    try
    {
        std::ostringstream oss;
        oss << "frame=" << _current_frame_index.load();
        oss << ",total=" << _frame_files.size();
        oss << ",state=" << static_cast<int>(_playback_state.load());

        _status_queue->push(oss.str());  // Send status update
    }
    catch (const std::exception&)
    {
        // Ignore status update failures
    }
}

void DCA1000Recording::validateFrameFile(const std::string& filepath)
{
    if (!std::filesystem::exists(filepath))
    {
        throw FileNotFoundException("Frame file does not exist: " + filepath);
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw FileNotFoundException("Cannot open frame file: " + filepath);
    }

    // Check if file is empty
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    if (file_size == 0)
    {
        throw InvalidFrameException("Empty frame file: " + filepath);
    }

    // Calculate expected file size based on ADC parameters
    if (_adc_params)
    {
        size_t expected_samples = _adc_params->chirps * _adc_params->tx *
                                  _adc_params->rx * _adc_params->samples *
                                  _adc_params->IQ;
        size_t expected_bytes = expected_samples * _adc_params->bytes;

        std::cout << "  File: "
                  << std::filesystem::path(filepath).filename().string()
                  << " - Size: " << file_size
                  << " bytes (expected: " << expected_bytes << " bytes)";

        if (file_size != expected_bytes)
        {
            std::cout << " - SIZE MISMATCH!" << std::endl;
            throw InvalidFrameException(
                "Frame file size mismatch: " + filepath + " (got " +
                std::to_string(file_size) + " bytes, expected " +
                std::to_string(expected_bytes) + " bytes)");
        }
        else
        {
            std::cout << " - OK" << std::endl;
        }
    }
}

void DCA1000Recording::validateDestDirectory()
{
    if (!std::filesystem::exists(_dest_dir))
    {
        throw DirectoryException("Destination directory does not exist: " +
                                 _dest_dir);
    }

    if (!std::filesystem::is_directory(_dest_dir))
    {
        throw DirectoryException("Path is not a directory: " + _dest_dir);
    }
}

}  // namespace radar
