#include <iostream>
#include <sstream>
#include <utility>

#include "frame_queue.h"

namespace sample
{

namespace
{

std::mutex& consoleMutex()
{
    static std::mutex mutex;
    return mutex;
}

} // namespace

void logLine(const std::string& line)
{
    std::lock_guard<std::mutex> lock(consoleMutex());
    std::cout << line << "\n";
}

void logError(const std::string& line)
{
    std::lock_guard<std::mutex> lock(consoleMutex());
    std::cerr << line << "\n";
}

void FrameQueue::push(FrameInfo frame)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_frames.push(std::move(frame));
    }
    m_frameReady.notify_one();
}

std::optional<FrameInfo> FrameQueue::pop()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_frameReady.wait(lock, [this] { return !m_frames.empty() || m_isShutDown; });
    if (m_frames.empty())
        return std::nullopt;

    FrameInfo frame = std::move(m_frames.front());
    m_frames.pop();
    return frame;
}

void FrameQueue::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_isShutDown = true;
    }
    m_frameReady.notify_all();
}

FrameWorkerPool::FrameWorkerPool(FrameQueue& queue, int workerCount) : m_queue(queue)
{
    for (int i = 0; i < workerCount; ++i)
    {
        m_workers.emplace_back([this] {
            while (std::optional<FrameInfo> frame = m_queue.pop())
            {
                std::ostringstream line;
                line << "camera " << frame->serial << " [" << frame->cameraIndex << "] frame " << frame->frameNumber
                     << " t=" << frame->timestampS << " s " << frame->width << "x" << frame->height
                     << " first pixel " << frame->firstPixel;
                logLine(line.str());
            }
        });
    }
}

FrameWorkerPool::~FrameWorkerPool()
{
    finish();
}

void FrameWorkerPool::finish()
{
    m_queue.shutdown();
    for (std::thread& worker : m_workers)
    {
        if (worker.joinable())
            worker.join();
    }
}

} // namespace sample
