/*
 * Sample name: Capture-50-images-zero-copy
 * Category: GenTL / zero-copy acquisition
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: GenTL
 * Short description: Configures the camera with xiAPI, then captures 50
 * images through the camera's GenTL data stream into buffers allocated by the
 * application (or by the producer with --auto-buffers) and prints the
 * metadata of each frame.
 */

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// xiApi.h pulls in <windows.h>, so it must come before the SDK GenTL helper
// headers: they declare `using namespace std;`, which makes the Windows `byte`
// type ambiguous with std::byte in C++17.
#include <xiApi.h>
#include <GenTLStream.h>
#include <xi_ImgBuffer.h>

static constexpr int frameCount = 50;
static constexpr int bufferCount = 10;
static constexpr int exposureUs = 10000;
static constexpr std::uint64_t frameTimeoutMs = 5000;
// PCIe transport DMA requires page-aligned buffers.
static constexpr std::size_t bufferAlignment = 4096;

struct Options
{
    bool shouldProducerAllocate = false;
};

static void checkXi(XI_RETURN status, const char* p_call)
{
    if (status != XI_OK)
        throw std::runtime_error(std::string(p_call) + " returned xiAPI error " + std::to_string(status));
}

// Opens the first camera and closes it when the object goes out of scope.
class XiCamera
{
public:
    XiCamera()
    {
        DWORD cameraCount = 0;
        checkXi(xiGetNumberDevices(&cameraCount), "xiGetNumberDevices");
        if (cameraCount == 0)
            throw std::runtime_error("no XIMEA cameras detected");
        checkXi(xiOpenDevice(0, &m_handle), "xiOpenDevice");
    }

    ~XiCamera()
    {
        const XI_RETURN status = xiCloseDevice(m_handle);
        if (status != XI_OK)
            std::cerr << "Warning: xiCloseDevice returned xiAPI error " << status << "\n";
    }

    XiCamera(const XiCamera&) = delete;
    XiCamera& operator=(const XiCamera&) = delete;

    HANDLE handle() const { return m_handle; }

private:
    HANDLE m_handle = nullptr;
};

// Runs a cleanup action when the scope ends, also during stack unwinding.
// The GenTL helpers throw on failure, and a destructor must not, so failures
// are reported on stderr instead.
class ScopeGuard
{
public:
    explicit ScopeGuard(std::function<void()> cleanup) : m_cleanup(std::move(cleanup)) {}
    ~ScopeGuard()
    {
        try
        {
            m_cleanup();
        }
        catch (const std::exception& error)
        {
            std::cerr << "Warning: cleanup failed: " << error.what() << "\n";
        }
    }

    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;

private:
    std::function<void()> m_cleanup;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--auto-buffers]\n"
              << "  --auto-buffers  Let the GenTL producer allocate the acquisition buffers.\n"
              << "                  By default the application allocates them and the camera\n"
              << "                  writes frames straight into application memory.\n";
}

static std::optional<Options> parseArguments(int argc, char* argv[], bool& isHelpRequested)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h")
        {
            isHelpRequested = true;
            return std::nullopt;
        }
        if (argument == "--auto-buffers")
            options.shouldProducerAllocate = true;
        else
        {
            std::cerr << "Error: unknown argument '" << argument << "'\n";
            return std::nullopt;
        }
    }
    return options;
}

static void printBufferInfo(GenTL::DS_HANDLE streamHandle, GenTL::BUFFER_HANDLE bufferHandle)
{
    ImgBuffer buffer(streamHandle);
    *buffer.GetHandlePtr() = bufferHandle;

    if (!buffer.HasNewData())
    {
        std::cerr << "Warning: frame transfer was interrupted, the buffer holds no valid image\n";
        return;
    }

    std::uint64_t frameId = 0;
    std::size_t width = 0;
    std::size_t height = 0;
    std::uint32_t exposureTimeUs = 0;
    std::uint64_t timestampNs = 0;
    void* p_data = nullptr;
    buffer.GetInfoUint64(GenTL::BUFFER_INFO_FRAMEID, &frameId);
    buffer.GetInfoSizet(GenTL::BUFFER_INFO_WIDTH, &width);
    buffer.GetInfoSizet(GenTL::BUFFER_INFO_HEIGHT, &height);
    buffer.GetInfoUint32(BUFFER_INFO_EXPOSURE_TIME_US, &exposureTimeUs);
    buffer.GetInfoUint64(GenTL::BUFFER_INFO_TIMESTAMP, &timestampNs);
    buffer.GetInfoPtr(GenTL::BUFFER_INFO_BASE, &p_data);

    const unsigned firstByte = p_data != nullptr ? *static_cast<const std::uint8_t*>(p_data) : 0;
    std::cout << "Frame " << std::setw(3) << frameId << "  " << width << "x" << height << "  exposure "
              << std::fixed << std::setprecision(3) << exposureTimeUs / 1000.0 << " ms  timestamp "
              << std::setprecision(6) << timestampNs / 1e9 << " s  first byte 0x" << std::hex << std::setw(2)
              << std::setfill('0') << firstByte << std::dec << std::setfill(' ') << "\n";
}

static int run(const Options& options)
{
    std::cout << "Opening first camera\n";
    XiCamera camera;
    const HANDLE cameraHandle = camera.handle();

    // XI_FRM_TRANSPORT_DATA delivers the frames as the camera sends them, with
    // no conversion on the host, which is what GenTL buffers carry.
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8), "xiSetParamInt(IMAGE_DATA_FORMAT)");
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_IMAGE_DATA_FORMAT, XI_FRM_TRANSPORT_DATA),
            "xiSetParamInt(IMAGE_DATA_FORMAT)");
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_EXPOSURE, exposureUs), "xiSetParamInt(EXPOSURE)");
    int payloadSize = 0;
    checkXi(xiGetParamInt(cameraHandle, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize), "xiGetParamInt(IMAGE_PAYLOAD_SIZE)");

    // From here on xiAPI only configures the camera; buffers and acquisition
    // run through the GenTL data stream that xiAPI hands out.
    checkXi(xiSetParamInt(cameraHandle, XI_PRM_GENTL_DATASTREAM_ENABLED, XI_ON),
            "xiSetParamInt(GENTL_DATASTREAM_ENABLED)");
    GenTL::DS_HANDLE streamHandle = nullptr;
    checkXi(xiGetParamString(cameraHandle, XI_PRM_GENTL_DATASTREAM_CONTEXT, &streamHandle, sizeof(streamHandle)),
            "xiGetParamString(GENTL_DATASTREAM_CONTEXT)");

    GenTLStream stream;
    stream.SetStreamHandle(streamHandle);
    stream.RegisterEvent(GenTL::EVENT_NEW_BUFFER);
    ScopeGuard unregisterEvent([&] { stream.UnregisterEvent(GenTL::EVENT_NEW_BUFFER); });

    // Declared before the guard that revokes the buffers, so the memory
    // outlives its use by the producer.
    std::vector<std::vector<std::uint8_t>> storage;
    std::vector<GenTL::BUFFER_HANDLE> buffers;
    ScopeGuard revokeBuffers([&] {
        stream.FlushQueue(GenTL::ACQ_QUEUE_ALL_DISCARD);
        for (GenTL::BUFFER_HANDLE buffer : buffers)
            stream.RevokeBuffer(buffer, nullptr);
    });

    std::cout << "Announcing " << bufferCount << " buffers of " << payloadSize << " bytes, allocated by the "
              << (options.shouldProducerAllocate ? "GenTL producer" : "application") << "\n";
    for (int i = 0; i < bufferCount; ++i)
    {
        GenTL::BUFFER_HANDLE buffer = GENTL_INVALID_HANDLE;
        if (options.shouldProducerAllocate)
        {
            stream.AllocAndAnnounceBuffer(static_cast<std::size_t>(payloadSize), &buffer);
        }
        else
        {
            // Over-allocate so std::align can place an aligned payload inside.
            std::vector<std::uint8_t>& bytes = storage.emplace_back(payloadSize + bufferAlignment);
            void* p_aligned = bytes.data();
            std::size_t space = bytes.size();
            std::align(bufferAlignment, static_cast<std::size_t>(payloadSize), p_aligned, space);
            stream.AnnounceBuffer(p_aligned, static_cast<std::size_t>(payloadSize), &buffer);
        }
        buffers.push_back(buffer);
        stream.QueueBuffer(buffer);
    }

    std::cout << "Starting acquisition\n";
    stream.StartAcquisition();
    ScopeGuard stopAcquisition([&] { stream.StopAcquisition(); });

    for (int i = 0; i < frameCount; ++i)
    {
        GenTL::EVENT_NEW_BUFFER_DATA eventData{};
        std::size_t eventSize = sizeof(eventData);
        stream.EventGetData(&eventData, &eventSize, frameTimeoutMs);
        printBufferInfo(streamHandle, eventData.BufferHandle);
        stream.QueueBuffer(eventData.BufferHandle);
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
    bool isHelpRequested = false;
    const std::optional<Options> options = parseArguments(argc, argv, isHelpRequested);
    if (!options)
    {
        printUsage(argv[0]);
        return isHelpRequested ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    try
    {
        return run(*options);
    }
    catch (const std::exception& error)
    {
        // GenTL_Exception from the SDK helpers also derives from std::exception.
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
