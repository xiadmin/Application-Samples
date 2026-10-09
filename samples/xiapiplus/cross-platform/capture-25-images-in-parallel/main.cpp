/*
 * Sample name: Capture-25-images-in-parallel
 * Category: Multi-camera / hardware trigger
 * OS platform: Cross-platform
 * Hardware platform: Two or more XIMEA cameras with GPIO, master GPO1 wired
 * to slave GPI1
 * API type: xiAPIplus
 * Short description: Opens all connected cameras in parallel, triggers a
 * master camera by software and the slaves through its GPO1 exposure pulse,
 * and captures 5 bursts of 5 frames per camera on per-camera acquisition
 * threads with a shared worker pool.
 */

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <xiapiplus.h>

#include "frame_queue.h"
#include "synchronized_camera.h"

static constexpr float exposureUs = 5000.0f;
static constexpr int acquisitionBufferBytes = 512 * 1024 * 1024;
static constexpr int queuedImageCount = 5;
static constexpr int workerCount = 4;
static constexpr auto burstInterval = std::chrono::milliseconds(1000);
// Slaves must be armed before the master can emit its first pulse.
static constexpr auto slaveArmDelay = std::chrono::milliseconds(50);

struct Options
{
    std::string masterSerial;
    int burstCount = 5;
    int burstLength = 5;
    int totalBandwidthMbps = 19200;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program
              << " [--master <serial>] [--bursts <n>] [--burst-length <n>] [--total-bandwidth <Mbps>]\n"
              << "  --master           Serial number of the master camera. Default: first camera found.\n"
              << "  --bursts           Software triggers sent to the master. Default: 5.\n"
              << "  --burst-length     Frames per trigger on every camera. Default: 5.\n"
              << "  --total-bandwidth  Host bandwidth shared equally by all cameras. Default: 19200.\n";
}

static bool parsePositiveInt(const std::string& text, int& value)
{
    try
    {
        std::size_t parsedLength = 0;
        value = std::stoi(text, &parsedLength);
        return parsedLength == text.size() && value > 0;
    }
    catch (const std::exception&)
    {
        return false;
    }
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
        if (i + 1 >= argc)
        {
            std::cerr << "Error: unknown or incomplete argument '" << argument << "'\n";
            return std::nullopt;
        }

        const std::string value = argv[++i];
        bool isValid = true;
        if (argument == "--master")
            options.masterSerial = value;
        else if (argument == "--bursts")
            isValid = parsePositiveInt(value, options.burstCount);
        else if (argument == "--burst-length")
            isValid = parsePositiveInt(value, options.burstLength);
        else if (argument == "--total-bandwidth")
            isValid = parsePositiveInt(value, options.totalBandwidthMbps);
        else
            isValid = false;

        if (!isValid)
        {
            std::cerr << "Error: invalid argument '" << argument << " " << value << "'\n";
            return std::nullopt;
        }
    }
    return options;
}

static void checkXi(XI_RETURN status, const char* p_call)
{
    if (status != XI_OK)
        throw std::runtime_error(std::string(p_call) + " returned xiAPI error " + std::to_string(status));
}

// Runs action on every camera in its own thread and waits for all of them.
// Opening and configuring several cameras concurrently saves start-up time.
// The first failure is rethrown after every thread has finished.
template <typename Action>
static void runOnAllCameras(std::vector<std::unique_ptr<sample::SynchronizedCamera>>& cameras, Action action)
{
    std::vector<std::future<void>> results;
    for (auto& p_camera : cameras)
        results.push_back(std::async(std::launch::async, [&action, &p_camera] { action(*p_camera); }));
    for (auto& result : results)
        result.wait();
    for (auto& result : results)
        result.get();
}

static int run(const Options& options)
{
    // The bandwidth is split by hand below, so xiAPI must not calculate it.
    // This is a global setting and applies to cameras opened afterwards.
    checkXi(xiSetParamInt(nullptr, XI_PRM_AUTO_BANDWIDTH_CALCULATION, XI_OFF),
            "xiSetParamInt(AUTO_BANDWIDTH_CALCULATION)");

    DWORD cameraCount = 0;
    checkXi(xiGetNumberDevices(&cameraCount), "xiGetNumberDevices");
    if (cameraCount == 0)
    {
        std::cerr << "Error: no XIMEA cameras detected\n";
        return EXIT_FAILURE;
    }
    if (cameraCount == 1)
        std::cerr << "Warning: only one camera detected; it runs as master without slaves\n";

    std::vector<std::string> serials;
    for (DWORD i = 0; i < cameraCount; ++i)
    {
        char serial[64] = "";
        checkXi(xiGetDeviceInfoString(i, XI_PRM_DEVICE_SN, serial, sizeof(serial)), "xiGetDeviceInfoString");
        serials.emplace_back(serial);
    }
    const std::string masterSerial = options.masterSerial.empty() ? serials.front() : options.masterSerial;

    // Declared first so it outlives the worker pool and the acquisition threads.
    sample::FrameQueue queue;
    sample::FrameWorkerPool workers(queue, workerCount);

    bool hasMaster = false;
    std::vector<std::unique_ptr<sample::SynchronizedCamera>> cameras;
    for (std::size_t i = 0; i < serials.size(); ++i)
    {
        const bool isMaster = serials[i] == masterSerial;
        hasMaster = hasMaster || isMaster;
        cameras.push_back(std::make_unique<sample::SynchronizedCamera>(
            serials[i], static_cast<int>(i), isMaster ? sample::CameraRole::MASTER : sample::CameraRole::SLAVE));
        std::cout << "  [" << i << "] " << serials[i] << (isMaster ? " (master)" : " (slave)") << "\n";
    }
    if (!hasMaster)
    {
        std::cerr << "Error: master camera " << masterSerial << " is not connected\n";
        return EXIT_FAILURE;
    }

    const auto openStart = std::chrono::steady_clock::now();
    runOnAllCameras(cameras, [](sample::SynchronizedCamera& camera) { camera.open(); });
    const auto openMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - openStart).count();
    std::cout << "Opened " << cameras.size() << " cameras in " << openMs << " ms\n";

    sample::CaptureSettings settings;
    settings.bandwidthLimitMbps = options.totalBandwidthMbps / static_cast<int>(cameras.size());
    settings.burstLength = options.burstLength;
    settings.exposureUs = exposureUs;
    settings.acquisitionBufferBytes = acquisitionBufferBytes;
    settings.queuedImageCount = queuedImageCount;
    std::cout << "Bandwidth limit per camera: " << settings.bandwidthLimitMbps << " Mbps\n";
    runOnAllCameras(cameras, [&settings](sample::SynchronizedCamera& camera) { camera.configure(settings); });

    sample::SynchronizedCamera* p_master = nullptr;
    for (auto& p_camera : cameras)
    {
        if (p_camera->isMaster())
            p_master = p_camera.get();
        else
            p_camera->startAcquisition(queue);
    }
    std::this_thread::sleep_for(slaveArmDelay);
    p_master->startAcquisition(queue);

    for (int i = 0; i < options.burstCount; ++i)
    {
        sample::logLine("--- burst " + std::to_string(i + 1) + "/" + std::to_string(options.burstCount) + " ---");
        p_master->trigger();
        std::this_thread::sleep_for(burstInterval);
    }

    for (auto& p_camera : cameras)
        p_camera->stopAcquisition();
    workers.finish();

    const std::uint64_t expectedCount = static_cast<std::uint64_t>(options.burstCount) * options.burstLength;
    bool isComplete = true;
    for (auto& p_camera : cameras)
    {
        const int skippedCount = p_camera->closeAndCountSkippedFrames();
        const std::uint64_t grabbedCount = p_camera->grabbedCount();
        std::cout << "Camera " << p_camera->serial() << ": grabbed " << grabbedCount << " of " << expectedCount
                  << ", skipped on transport " << skippedCount << "\n";
        isComplete = isComplete && !p_camera->hasFailed() && grabbedCount == expectedCount && skippedCount == 0;
    }

    if (!isComplete)
    {
        std::cerr << "Error: not every camera captured all frames; check the GPO1 -> GPI1 wiring and bandwidth\n";
        return EXIT_FAILURE;
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
    catch (xiAPIplus_Exception& exception)
    {
        std::cerr << "Error: camera operation failed: " << sample::describeXiError(exception) << "\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
