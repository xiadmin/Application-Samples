#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace sample
{

// Metadata of one captured frame, handed from an acquisition thread to the
// worker pool. The pixel data stays in the camera buffer.
struct FrameInfo
{
    std::string serial;
    int cameraIndex = 0;
    int frameNumber = 0;
    double timestampS = 0.0;
    int width = 0;
    int height = 0;
    unsigned firstPixel = 0;
};

// Thread-safe FIFO between the acquisition threads and the worker threads.
class FrameQueue
{
public:
    void push(FrameInfo frame);
    // Blocks until a frame is available. Returns std::nullopt once the queue
    // is shut down and empty.
    std::optional<FrameInfo> pop();
    void shutdown();

private:
    std::mutex m_mutex;
    std::condition_variable m_frameReady;
    std::queue<FrameInfo> m_frames;
    bool m_isShutDown = false;
};

// Worker threads that drain a FrameQueue and print each frame. A real
// application would process the image here. The destructor shuts the queue
// down and joins the threads, also when an exception unwinds the stack.
class FrameWorkerPool
{
public:
    FrameWorkerPool(FrameQueue& queue, int workerCount);
    ~FrameWorkerPool();

    FrameWorkerPool(const FrameWorkerPool&) = delete;
    FrameWorkerPool& operator=(const FrameWorkerPool&) = delete;

    // Processes the remaining frames, then stops the workers.
    void finish();

private:
    FrameQueue& m_queue;
    std::vector<std::thread> m_workers;
};

// Writes one line to stdout or stderr without interleaving with other threads.
void logLine(const std::string& line);
void logError(const std::string& line);

} // namespace sample
