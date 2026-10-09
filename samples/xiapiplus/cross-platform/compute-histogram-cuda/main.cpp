/*
 * Sample name: Compute-histogram-cuda
 * Category: GPU processing / CUDA integration
 * OS platform: Cross-platform
 * Hardware platform: NVIDIA GPU, GPUDirect RDMA target needs a XIMEA PCIe
 * camera
 * API type: xiAPIplus
 * Short description: Delivers 8-bit frames to CPU, CUDA managed, mapped host
 * or GPU memory with XI_PRM_TRANSPORT_DATA_TARGET, computes a 64-bin
 * histogram of each frame with a CUDA kernel and prints it with the
 * processing time and throughput.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <xiapiplus.h>

#include "cuda_helpers.h"
#include "histogram64.h"

// Where xiAPI places the image data, set with XI_PRM_TRANSPORT_DATA_TARGET.
enum class MemoryTarget
{
    CPU,      // ordinary host memory; the sample copies each frame with cudaMemcpy
    UNIFIED,  // CUDA managed memory, read by the GPU directly
    ZEROCOPY, // mapped (pinned) host memory, read by the GPU over the bus
    GPU       // GPU memory written by the camera over GPUDirect RDMA
};

struct Options
{
    MemoryTarget target = MemoryTarget::CPU;
    int frameCount = 100;
};

static constexpr float exposureUs = 5000.0f;
static constexpr int frameTimeoutMs = 5000;
// GPUDirect RDMA writes into GPU memory exposed through the PCIe BAR, which is
// often only a few hundred MB, so the acquisition buffer holds 4 frames.
static constexpr int gpuBufferFrameCount = 4;
// Darkest to brightest shade used to draw one histogram bin.
static const std::string histogramShades = " .',;\"oO%8@#";

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--target cpu|unified|zerocopy|gpu] [--frame-count <n>]\n"
              << "  --target       Memory xiAPI delivers the frames to. Default: cpu.\n"
              << "                 cpu      host memory, copied to the GPU with cudaMemcpy\n"
              << "                 unified  CUDA managed memory\n"
              << "                 zerocopy mapped host memory read by the GPU directly\n"
              << "                 gpu      GPU memory over GPUDirect RDMA (PCIe cameras on Linux)\n"
              << "  --frame-count  Frames to process after one warm-up frame. Default: 100.\n";
}

static std::optional<MemoryTarget> parseTarget(const std::string& text)
{
    if (text == "cpu")
        return MemoryTarget::CPU;
    if (text == "unified")
        return MemoryTarget::UNIFIED;
    if (text == "zerocopy")
        return MemoryTarget::ZEROCOPY;
    if (text == "gpu")
        return MemoryTarget::GPU;
    return std::nullopt;
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
        if (argument == "--target")
        {
            const std::optional<MemoryTarget> target = parseTarget(value);
            isValid = target.has_value();
            if (isValid)
                options.target = *target;
        }
        else if (argument == "--frame-count")
            isValid = parsePositiveInt(value, options.frameCount);
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

static std::string describeXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::ostringstream text;
    text << "xiAPI error " << exception.GetErrorNumber() << ": " << description;
    return text.str();
}

static XI_TRANSPORT_DATA_TARGET_MODE toXiTarget(MemoryTarget target)
{
    switch (target)
    {
    case MemoryTarget::UNIFIED:
        return XI_TRANSPORT_DATA_TARGET_UNIFIED;
    case MemoryTarget::ZEROCOPY:
        return XI_TRANSPORT_DATA_TARGET_ZEROCOPY;
    case MemoryTarget::GPU:
        return XI_TRANSPORT_DATA_TARGET_GPU_RAM;
    case MemoryTarget::CPU:
        break;
    }
    return XI_TRANSPORT_DATA_TARGET_CPU_RAM;
}

// Stops the acquisition and closes the camera when the scope ends, so that the
// xiAPIplus_Camera destructor, which cannot report errors, finds it closed.
class CameraCloser
{
public:
    explicit CameraCloser(xiAPIplus_Camera& camera) : m_camera(camera) {}
    ~CameraCloser()
    {
        try
        {
            if (m_camera.IsAcquisitionActive())
                m_camera.StopAcquisition();
            m_camera.Close();
        }
        catch (xiAPIplus_Exception& exception)
        {
            std::cerr << "Warning: closing the camera failed: " << describeXiError(exception) << "\n";
        }
    }

    CameraCloser(const CameraCloser&) = delete;
    CameraCloser& operator=(const CameraCloser&) = delete;

private:
    xiAPIplus_Camera& m_camera;
};

// Selects CUDA device 0 and checks that it supports the requested memory
// target. Returns the device properties.
static cudaDeviceProp initCudaDevice(MemoryTarget target)
{
    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess || deviceCount == 0)
        throw std::runtime_error("no CUDA device found; this sample needs an NVIDIA GPU and its driver");

    // Mapped host memory must be enabled before the CUDA context is created,
    // which happens on the first runtime call that needs it.
    sample::checkCuda(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");
    sample::checkCuda(cudaSetDevice(0), "cudaSetDevice");

    cudaDeviceProp properties{};
    sample::checkCuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    std::cout << "CUDA device 0: " << properties.name << "\n";

    if (target != MemoryTarget::CPU && !properties.unifiedAddressing)
        throw std::runtime_error("the CUDA device does not support unified addressing; use --target cpu");
    if (target == MemoryTarget::UNIFIED && (!properties.managedMemory || !properties.concurrentManagedAccess))
    {
        throw std::runtime_error("the CUDA device does not support concurrent managed memory access, which "
                                 "--target unified needs (it is not available on Windows); use --target cpu");
    }
    if (target == MemoryTarget::ZEROCOPY && !properties.canMapHostMemory)
        throw std::runtime_error("the CUDA device cannot map host memory; use --target cpu");
    return properties;
}

static void configureCamera(xiAPIplus_Camera& camera, MemoryTarget target)
{
    camera.SetExposureTime(exposureUs);
    // xiAPI cannot post-process data it writes outside host memory, so the
    // frames stay in their 8-bit transport format: MONO8 on monochrome cameras,
    // RAW8 (Bayer) on color ones.
    camera.SetImageDataFormat(XI_FRM_TRANSPORT_DATA);
    camera.SetDeviceOutputDataBitDepth(XI_BPP_8);

    try
    {
        camera.SetTransportDataTarget(toXiTarget(target));
    }
    catch (xiAPIplus_Exception& exception)
    {
        if (exception.GetErrorNumber() == XI_GPUDIRECT_NOT_AVAILABLE)
        {
            throw std::runtime_error("GPUDirect RDMA is not available (XI_GPUDIRECT_NOT_AVAILABLE). --target gpu "
                                     "needs a XIMEA PCIe camera, Linux, a GPU that supports GPUDirect and the "
                                     "XIMEA driver built with GPUDirect support; use another --target");
        }
        throw;
    }

    if (target == MemoryTarget::GPU)
        camera.SetAcquisitionBufferSizeBytes(camera.GetImagePayloadSize() * gpuBufferFrameCount);
}

// Draws the 64 bins as one line of characters, darker for fewer pixels. The
// scale is logarithmic so that small bins stay visible next to large ones.
static std::string drawHistogram(const std::vector<unsigned int>& histogram, std::size_t pixelCount)
{
    const std::size_t shadeCount = histogramShades.size();
    const double maxLevel = std::log1p(static_cast<double>(pixelCount));
    std::string line;
    for (unsigned int count : histogram)
    {
        const auto shade = static_cast<std::size_t>(std::log1p(static_cast<double>(count)) / maxLevel * shadeCount);
        line += histogramShades[std::min(shade, shadeCount - 1)];
    }
    return line;
}

static int run(const Options& options)
{
    xiAPIplus_Camera camera;
    if (camera.GetNumberOfConnectedCameras() == 0)
    {
        std::cerr << "Error: no XIMEA cameras detected\n";
        return EXIT_FAILURE;
    }

    // CUDA is set up before the camera, because xiAPI allocates managed and
    // mapped memory in the CUDA context that this creates.
    initCudaDevice(options.target);
    int canUseHostPointer = 0;
    sample::checkCuda(cudaDeviceGetAttribute(&canUseHostPointer, cudaDevAttrCanUseHostPointerForRegisteredMem, 0),
                      "cudaDeviceGetAttribute(CanUseHostPointerForRegisteredMem)");

    std::cout << "Opening first camera\n";
    camera.OpenFirst();
    CameraCloser closer(camera);
    configureCamera(camera, options.target);
    camera.SetNextImageTimeout_ms(frameTimeoutMs);

    const int width = camera.GetWidth();
    const int height = camera.GetHeight();
    const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
    std::cout << "Image size " << width << "x" << height << ", payload " << camera.GetImagePayloadSize()
              << " bytes\n";

    sample::DeviceBuffer<unsigned char> copyBuffer;
    if (options.target == MemoryTarget::CPU)
        copyBuffer = sample::allocateDeviceBuffer<unsigned char>(pixelCount);
    sample::DeviceBuffer<unsigned int> deviceHistogram =
        sample::allocateDeviceBuffer<unsigned int>(sample::histogramBinCount);
    std::vector<unsigned int> histogram(sample::histogramBinCount);

    camera.StartAcquisition();
    std::cout << "Computing a 64-bin GPU histogram of " << pixelCount << " bytes for " << options.frameCount
              << " frames\n";

    xiAPIplus_Image image;
    double totalSeconds = 0.0;
    // Frame -1 warms up the GPU and is not timed.
    for (int i = -1; i < options.frameCount; ++i)
    {
        camera.GetNextImage(&image);
        auto* p_frame = static_cast<unsigned char*>(image.GetXI_IMG()->bp);

        sample::checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto start = std::chrono::steady_clock::now();

        const unsigned char* p_gpuFrame = p_frame;
        if (options.target == MemoryTarget::CPU)
        {
            sample::checkCuda(cudaMemcpy(copyBuffer.get(), p_frame, pixelCount, cudaMemcpyHostToDevice),
                              "cudaMemcpy(frame)");
            p_gpuFrame = copyBuffer.get();
        }
        else if (options.target == MemoryTarget::ZEROCOPY && !canUseHostPointer)
        {
            // Some platforms address mapped host memory through a separate
            // device pointer instead of the host pointer itself.
            void* p_mapped = nullptr;
            sample::checkCuda(cudaHostGetDevicePointer(&p_mapped, p_frame, 0), "cudaHostGetDevicePointer");
            p_gpuFrame = static_cast<const unsigned char*>(p_mapped);
        }

        sample::computeHistogram64(p_gpuFrame, pixelCount, deviceHistogram.get());
        sample::checkCuda(cudaMemcpy(histogram.data(), deviceHistogram.get(),
                                     sample::histogramBinCount * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                          "cudaMemcpy(histogram)");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (i < 0)
            continue;

        std::size_t countedPixels = 0;
        for (unsigned int count : histogram)
            countedPixels += count;
        if (countedPixels != pixelCount)
        {
            std::cerr << "Error: the histogram of frame " << i + 1 << " counts " << countedPixels << " of "
                      << pixelCount << " pixels\n";
            return EXIT_FAILURE;
        }

        totalSeconds += seconds;
        char timing[32];
        std::snprintf(timing, sizeof(timing), "%.3f ms", seconds * 1e3);
        std::cout << "Frame " << i + 1 << "/" << options.frameCount << " |" << drawHistogram(histogram, pixelCount)
                  << "| " << timing << "\n";
    }

    camera.StopAcquisition();
    camera.SetCounterSelector(XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES);
    const int transportSkipped = camera.GetCounterValue();
    camera.SetCounterSelector(XI_CNT_SEL_API_SKIPPED_FRAMES);
    const int apiSkipped = camera.GetCounterValue();

    const double averageSeconds = totalSeconds / options.frameCount;
    char summary[128];
    std::snprintf(summary, sizeof(summary), "Average copy + histogram time: %.3f ms, %.2f MB/s",
                  averageSeconds * 1e3, pixelCount * 1e-6 / averageSeconds);
    std::cout << summary << "\n"
              << "Skipped frames: transport " << transportSkipped << ", API " << apiSkipped << "\n"
              << "Done\n";
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
        std::cerr << "Error: camera operation failed: " << describeXiError(exception) << "\n";
        return EXIT_FAILURE;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
