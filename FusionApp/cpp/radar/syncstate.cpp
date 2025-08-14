#include "syncstate.hpp"
#include <chrono>
#include <thread>

namespace radar
{

// Playback state operations
void SyncStateUtils::setPlaybackState(SyncState& sync_state,
                                      PlaybackState state)
{
    sync_state.playback_state.store(state);
}

PlaybackState SyncStateUtils::getPlaybackState(const SyncState& sync_state)
{
    return sync_state.playback_state.load();
}

// Timeline operations
void SyncStateUtils::setCurrentTimelinePosition(SyncState& sync_state,
                                                double position)
{
    sync_state.current_timeline_position.store(position);
}

double SyncStateUtils::getCurrentTimelinePosition(const SyncState& sync_state)
{
    return sync_state.current_timeline_position.load();
}

// Timestamp operations
void SyncStateUtils::setStartTimestamp(SyncState& sync_state, double timestamp)
{
    sync_state.start_timestamp.store(timestamp);
}

double SyncStateUtils::getStartTimestamp(const SyncState& sync_state)
{
    return sync_state.start_timestamp.load();
}

void SyncStateUtils::setLastRadarTimestamp(SyncState& sync_state,
                                           double timestamp)
{
    sync_state.last_radar_timestamp.store(timestamp);
}

double SyncStateUtils::getLastRadarTimestamp(const SyncState& sync_state)
{
    return sync_state.last_radar_timestamp.load();
}

// Feed synchronization
void SyncStateUtils::signalFeedReady(SyncState& sync_state)
{
    sync_state.feed_ready.store(true);
}

bool SyncStateUtils::isFeedReady(const SyncState& sync_state)
{
    return sync_state.feed_ready.load();
}

void SyncStateUtils::signalStart(SyncState& sync_state)
{
    sync_state.start_signal.store(true);
}

bool SyncStateUtils::waitForStartSignal(const SyncState& sync_state,
                                        double timeout_seconds)
{
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::duration<double>(timeout_seconds);

    while (!sync_state.start_signal.load())
    {
        auto current_time = std::chrono::steady_clock::now();
        if (current_time - start_time >= timeout_duration)
        {
            return false;  // Timeout
        }

        // Sleep briefly to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return true;
}

// Seeking operations
void SyncStateUtils::seekToTime(SyncState& sync_state, double time)
{
    std::lock_guard<std::mutex> lock(sync_state.control_mutex);
    sync_state.current_timeline_position.store(time);
}

// Reset operations
void SyncStateUtils::reset(SyncState& sync_state)
{
    std::lock_guard<std::mutex> lock(sync_state.control_mutex);

    sync_state.playback_state.store(PlaybackState::STOPPED);
    sync_state.current_timeline_position.store(0.0);
    sync_state.start_timestamp.store(0.0);
    sync_state.last_radar_timestamp.store(0.0);
    sync_state.feed_ready.store(false);
    sync_state.start_signal.store(false);
}

}  // namespace radar
