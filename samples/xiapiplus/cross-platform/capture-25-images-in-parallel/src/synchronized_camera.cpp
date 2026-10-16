#include <sstream>
#include <utility>

#include "synchronized_camera.h"

namespace sample
{

namespace
{

constexpr int frameTimeoutMs = 1000;
// After stopAcquisition() the loop keeps reading with a short timeout until
// the frames already in the queue are drained.
constexpr int drainTimeoutMs = 200;

} // namespace

std::string describeXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::ostringstream text;
    text << "xiAPI error " << exception.GetErrorNumber() << ": " << description;
    return text.str();
}

SynchronizedCamera::SynchronizedCamera(std::string serial, int index, CameraRole role)
    : m_serial(std::move(serial)), m_index(index), m_role(role)
{
}

SynchronizedCamera::~SynchronizedCamera()
{
    m_isRunning = false;
    if (m_thread.joinable())
        m_thread.join();

    // Close here so that the xiAPIplus_Camera destructor, which cannot report
    // errors, finds the camera already closed.
    try
    {
        m_camera.Close();
    }
    catch (xiAPIplus_Exception& exception)
    {
        logError("Warning: closing camera " + m_serial + " failed: " + describeXiError(exception));
    }
}

void SynchronizedCamera::open()
{
    m_camera.OpenBySN(m_serial.c_str());
}

void SynchronizedCamera::configure(const CaptureSettings& settings)
{
    // Every camera gets an equal share of the host bandwidth, so the group
    // does not overload the shared link when all cameras expose at once.
    m_camera.SetBandwidthLimitMode(XI_ON);
    m_camera.SetBandwidthLimit(settings.bandwidthLimitMbps);
    m_camera.SetImageDataFormat(XI_RAW8);
    m_camera.SetExposureTime(settings.exposureUs);
    m_camera.SetBufferPolicy(XI_BP_SAFE);
    m_camera.SetAcquisitionBufferSizeBytes(settings.acquisitionBufferBytes);
    m_camera.SetAcquisitionQueueImagesCount(settings.queuedImageCount);

    if (isMaster())
    {
        // One software trigger starts a burst of frames. GPO1 emits a pulse at
        // the start of every exposure; that pulse triggers the slaves.
        m_camera.SetTriggerSelector(XI_TRG_SEL_FRAME_BURST_START);
        m_camera.SetAcquisitionFrameBurstCount(settings.burstLength);
        m_camera.SetTriggerSource(XI_TRG_SOFTWARE);
        m_camera.SetGPOSelector(XI_GPO_PORT1);
        m_camera.SetGPOMode(XI_GPO_EXPOSURE_PULSE);
    }
    else
    {
        // Each rising edge on GPI1 starts one frame.
        m_camera.SetTriggerSelector(XI_TRG_SEL_FRAME_START);
        m_camera.SetGPISelector(XI_GPI_PORT1);
        m_camera.SetGPIMode(XI_GPI_TRIGGER);
        m_camera.SetTriggerSource(XI_TRG_EDGE_RISING);
    }

    // Not every camera model supports trigger overlap or timestamp reset. The
    // capture still works without them, so a failure is only reported.
    try
    {
        m_camera.SetTriggerOverlap(XI_TRG_OVERLAP_PREV_FRAME);
    }
    catch (xiAPIplus_Exception& exception)
    {
        logError("Warning: camera " + m_serial + " does not support trigger overlap: " + describeXiError(exception));
    }
    try
    {
        m_camera.SetTimestampResetSource(XI_TS_RST_SRC_SW);
        m_camera.SetTimestampResetMode(XI_TS_RST_ARM_ONCE);
    }
    catch (xiAPIplus_Exception& exception)
    {
        logError("Warning: camera " + m_serial + " does not support timestamp reset: " + describeXiError(exception));
    }

    logLine("Configured camera " + m_serial + (isMaster() ? " as master" : " as slave"));
}

void SynchronizedCamera::startAcquisition(FrameQueue& queue)
{
    m_camera.StartAcquisition();
    m_isRunning = true;
    m_thread = std::thread(&SynchronizedCamera::acquisitionLoop, this, std::ref(queue));
}

void SynchronizedCamera::trigger()
{
    m_camera.SetTriggerSoftware(1);
}

void SynchronizedCamera::stopAcquisition()
{
    m_isRunning = false;
    if (m_thread.joinable())
        m_thread.join();
    m_camera.StopAcquisition();
}

int SynchronizedCamera::closeAndCountSkippedFrames()
{
    m_camera.SetCounterSelector(XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES);
    const int skippedCount = m_camera.GetCounterValue();
    m_camera.Close();
    return skippedCount;
}

void SynchronizedCamera::acquisitionLoop(FrameQueue& queue)
{
    xiAPIplus_Image image;
    while (true)
    {
        const bool isStopping = !m_isRunning.load();
        m_camera.SetNextImageTimeout_ms(isStopping ? drainTimeoutMs : frameTimeoutMs);
        try
        {
            m_camera.GetNextImage(&image);
        }
        catch (xiAPIplus_Exception& exception)
        {
            const XI_RETURN status = exception.GetErrorNumber();
            // Slaves wait for the master's pulses, so timeouts are normal
            // between bursts. While stopping, a timeout means the queue is empty.
            if (status == XI_TIMEOUT && !isStopping)
                continue;
            if (status != XI_TIMEOUT && status != XI_ACQUISITION_STOPED)
            {
                logError("Error: camera " + m_serial + " image read failed: " + describeXiError(exception));
                m_hasFailed = true;
            }
            return;
        }

        ++m_grabbedCount;
        const XI_IMG* p_xiImage = image.GetXI_IMG();
        FrameInfo frame;
        frame.serial = m_serial;
        frame.cameraIndex = m_index;
        frame.frameNumber = static_cast<int>(p_xiImage->nframe);
        frame.timestampS = p_xiImage->tsSec + 1e-6 * p_xiImage->tsUSec;
        frame.width = image.GetWidth();
        frame.height = image.GetHeight();
        frame.firstPixel = image.GetPixels() != nullptr ? image.GetPixels()[0] : 0;
        queue.push(std::move(frame));
    }
}

} // namespace sample
