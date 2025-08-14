#pragma once

#include <atomic>
#include <mutex>

namespace radar
{

/**
 * Playback state enumeration for synchronized playback control
 */
enum class PlaybackState
{
    STOPPED,
    PLAYING,
    PAUSED
};

/**
 * Synchronized state structure for multithreaded radar processing
 * Uses atomic operations where possible for lock-free access
 */
struct SyncState
{
    // Playback control
    std::atomic<PlaybackState> playback_state{PlaybackState::STOPPED};
    std::atomic<double> current_timeline_position{0.0};
    std::atomic<double> start_timestamp{0.0};
    std::atomic<double> last_radar_timestamp{0.0};

    // Feed synchronization
    std::atomic<bool> feed_ready{false};
    std::atomic<bool> start_signal{false};

    // Mutex for complex operations that need synchronization
    mutable std::mutex control_mutex;

    // Constructor
    SyncState() = default;

    // Disable copy and move operations for safety with atomics and mutex
    SyncState(const SyncState&) = delete;
    SyncState& operator=(const SyncState&) = delete;
    SyncState(SyncState&&) = delete;
    SyncState& operator=(SyncState&&) = delete;
};

/**
 * Utility class for synchronized state operations
 */
class SyncStateUtils
{
  public:
    // Playback state operations
    static void setPlaybackState(SyncState& sync_state, PlaybackState state);
    static PlaybackState getPlaybackState(const SyncState& sync_state);

    // Timeline operations
    static void setCurrentTimelinePosition(SyncState& sync_state,
                                           double position);
    static double getCurrentTimelinePosition(const SyncState& sync_state);

    // Timestamp operations
    static void setStartTimestamp(SyncState& sync_state, double timestamp);
    static double getStartTimestamp(const SyncState& sync_state);

    static void setLastRadarTimestamp(SyncState& sync_state, double timestamp);
    static double getLastRadarTimestamp(const SyncState& sync_state);

    // Feed synchronization
    static void signalFeedReady(SyncState& sync_state);
    static bool isFeedReady(const SyncState& sync_state);

    static void signalStart(SyncState& sync_state);
    static bool waitForStartSignal(const SyncState& sync_state,
                                   double timeout_seconds = 30.0);

    // Seeking operations
    static void seekToTime(SyncState& sync_state, double time);

    // Reset operations
    static void reset(SyncState& sync_state);
};

}  // namespace radar
