#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include <xiapiplus.h>

#include "frame_queue.h"

namespace sample
{

enum class CameraRole
{
    MASTER, // started by a software trigger, drives the slaves through GPO1
    SLAVE   // exposes one frame per rising edge on GPI1
};

struct CaptureSettings
{
    int bandwidthLimitMbps = 0;
    int burstLength = 0;
    float exposureUs = 0.0f;
    int acquisitionBufferBytes = 0;
    int queuedImageCount = 0;
};

// One camera of a hardware-synchronized group, with its own acquisition
// thread. xiAPIplus_Camera closes the camera in its destructor; this class
// also stops and joins the acquisition thread first.
class SynchronizedCamera
{
public:
    SynchronizedCamera(std::string serial, int index, CameraRole role);
    ~SynchronizedCamera();

    SynchronizedCamera(const SynchronizedCamera&) = delete;
    SynchronizedCamera& operator=(const SynchronizedCamera&) = delete;

    void open();
    void configure(const CaptureSettings& settings);
    // Starts acquisition and a thread that pushes every frame into the queue.
    void startAcquisition(FrameQueue& queue);
    // Fires one software trigger; valid for the master only.
    void trigger();
    // Lets the acquisition thread drain the remaining frames and exit.
    void stopAcquisition();
    // Reads the transport skipped-frames counter and closes the camera.
    int closeAndCountSkippedFrames();

    const std::string& serial() const { return m_serial; }
    bool isMaster() const { return m_role == CameraRole::MASTER; }
    std::uint64_t grabbedCount() const { return m_grabbedCount.load(); }
    bool hasFailed() const { return m_hasFailed.load(); }

private:
    void acquisitionLoop(FrameQueue& queue);

    std::string m_serial;
    int m_index;
    CameraRole m_role;
    xiAPIplus_Camera m_camera;
    std::thread m_thread;
    std::atomic<bool> m_isRunning{false};
    std::atomic<bool> m_hasFailed{false};
    std::atomic<std::uint64_t> m_grabbedCount{0};
};

std::string describeXiError(xiAPIplus_Exception& exception);

} // namespace sample
