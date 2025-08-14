#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace radar
{

/**
 * Thread-safe queue implementation for radar frame processing
 * Uses RAII and provides blocking and non-blocking operations
 */
template <typename T>
class ThreadSafeQueue
{
  private:
    mutable std::mutex _mutex;
    std::queue<T> _queue;
    std::condition_variable _condition;

  public:
    ThreadSafeQueue() = default;

    // Disable copy operations for safety
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    /**
     * Add an item to the queue (thread-safe)
     * @param item The item to add
     */
    void push(const T& item)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(item);
        _condition.notify_one();
    }

    /**
     * Add an item to the queue using move semantics (thread-safe)
     * @param item The item to move into the queue
     */
    void push(T&& item)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _queue.push(std::move(item));
        _condition.notify_one();
    }

    /**
     * Try to pop an item from the queue without blocking
     * @param item Reference to store the popped item
     * @return true if an item was popped, false if queue was empty
     */
    bool tryPop(T& item)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.empty())
        {
            return false;
        }
        item = std::move(_queue.front());
        _queue.pop();
        return true;
    }

    /**
     * Try to pop an item from the queue without blocking
     * @return unique_ptr containing the item, or nullptr if queue was empty
     */
    std::unique_ptr<T> tryPop()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_queue.empty())
        {
            return nullptr;
        }
        auto result = std::make_unique<T>(std::move(_queue.front()));
        _queue.pop();
        return result;
    }

    /**
     * Wait for an item and pop it from the queue (blocking)
     * @param item Reference to store the popped item
     */
    void waitAndPop(T& item)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this] { return !_queue.empty(); });
        item = std::move(_queue.front());
        _queue.pop();
    }

    /**
     * Wait for an item and pop it from the queue (blocking)
     * @return unique_ptr containing the popped item
     */
    std::unique_ptr<T> waitAndPop()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this] { return !_queue.empty(); });
        auto result = std::make_unique<T>(std::move(_queue.front()));
        _queue.pop();
        return result;
    }

    /**
     * Wait for an item with timeout
     * @param item Reference to store the popped item
     * @param timeout_ms Timeout in milliseconds
     * @return true if item was popped, false if timeout occurred
     */
    template <typename Rep, typename Period>
    bool waitAndPop(T& item, const std::chrono::duration<Rep, Period>& timeout)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        if (_condition.wait_for(lock, timeout,
                                [this] { return !_queue.empty(); }))
        {
            item = std::move(_queue.front());
            _queue.pop();
            return true;
        }
        return false;
    }

    /**
     * Check if the queue is empty (thread-safe)
     * @return true if empty, false otherwise
     */
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.empty();
    }

    /**
     * Get the current size of the queue (thread-safe)
     * @return The number of items in the queue
     */
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _queue.size();
    }

    /**
     * Clear all items from the queue (thread-safe)
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        std::queue<T> empty_queue;
        _queue.swap(empty_queue);
    }
};

}  // namespace radar
