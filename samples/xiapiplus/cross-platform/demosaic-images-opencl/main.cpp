/*
 * Sample name: Demosaic-images-opencl
 * Category: GPU processing / OpenCL integration
 * OS platform: Cross-platform
 * Hardware platform: XIMEA color camera, OpenCL GPU or other OpenCL device
 * API type: xiAPIplus
 * Short description: Captures RAW8 Bayer frames from a color camera,
 * demosaics them with a Malvar-He-Cutler OpenCL kernel on the GPU, saves them
 * as RGB TIFF files and prints the kernel and transfer times.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <xiapiplus.h>

#include "opencl_demosaic.h"
#include "tiff_writer.h"

namespace fs = std::filesystem;

static constexpr int frameTimeoutMs = 5000;

struct Options
{
    int frameCount = 10;
    float exposureUs = 10000.0f;
    fs::path outputDir = ".";
};

// Position of the red pixel in the top-left 2x2 tile of a Bayer frame.
struct RedPixelOffset
{
    int column = 0;
    int row = 0;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--frame-count <n>] [--exposure-us <us>] [--output-dir <dir>]\n"
              << "  --frame-count  Frames to demosaic and save. Default: 10.\n"
              << "  --exposure-us  Exposure time in microseconds. Default: 10000.\n"
              << "  --output-dir   Directory for the RGB TIFF files. Default: current directory.\n";
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

static bool parsePositiveFloat(const std::string& text, float& value)
{
    try
    {
        std::size_t parsedLength = 0;
        value = std::stof(text, &parsedLength);
        return parsedLength == text.size() && value > 0.0f;
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
        if (argument == "--frame-count")
            isValid = parsePositiveInt(value, options.frameCount);
        else if (argument == "--exposure-us")
            isValid = parsePositiveFloat(value, options.exposureUs);
        else if (argument == "--output-dir")
            options.outputDir = value;
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

// Maps the camera's color filter array to the red pixel position the kernel
// needs. Only the four plain Bayer layouts are supported.
static std::optional<RedPixelOffset> redPixelOffsetFor(XI_COLOR_FILTER_ARRAY filterArray)
{
    switch (filterArray)
    {
    case XI_CFA_BAYER_RGGB:
        return RedPixelOffset{0, 0};
    case XI_CFA_BAYER_GRBG:
        return RedPixelOffset{1, 0};
    case XI_CFA_BAYER_GBRG:
        return RedPixelOffset{0, 1};
    case XI_CFA_BAYER_BGGR:
        return RedPixelOffset{1, 1};
    default:
        return std::nullopt;
    }
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

static int run(const Options& options)
{
    xiAPIplus_Camera camera;
    if (camera.GetNumberOfConnectedCameras() == 0)
    {
        std::cerr << "Error: no XIMEA cameras detected\n";
        return EXIT_FAILURE;
    }

    std::cout << "Opening first camera\n";
    camera.OpenFirst();
    CameraCloser closer(camera);
    if (!camera.IsSensorColor())
    {
        std::cerr << "Error: the camera is monochrome; this sample needs a color camera with a Bayer filter\n";
        return EXIT_FAILURE;
    }

    // RAW8 leaves the Bayer data unprocessed, so the GPU does the demosaicing
    // instead of xiAPI on the CPU.
    camera.SetImageDataFormat(XI_RAW8);
    camera.SetExposureTime(options.exposureUs);
    camera.SetNextImageTimeout_ms(frameTimeoutMs);

    const XI_COLOR_FILTER_ARRAY filterArray = camera.GetSensorColorFilterArray();
    const std::optional<RedPixelOffset> redOffset = redPixelOffsetFor(filterArray);
    if (!redOffset)
    {
        std::cerr << "Error: the camera reports color filter array " << filterArray
                  << ", which is not a Bayer RGGB, GRBG, GBRG or BGGR pattern\n";
        return EXIT_FAILURE;
    }

    // xiAPI does not apply white balance to RAW8 data, so the kernel applies
    // the camera's current coefficients.
    sample::ChannelGains gains;
    gains.red = camera.GetWhiteBalanceRed();
    gains.green = camera.GetWhiteBalanceGreen();
    gains.blue = camera.GetWhiteBalanceBlue();

    const int width = camera.GetWidth();
    const int height = camera.GetHeight();
    std::cout << "Image size " << width << "x" << height << ", color filter array " << filterArray
              << ", white balance R " << gains.red << " G " << gains.green << " B " << gains.blue << "\n";

    sample::OpenClDemosaic demosaic(width, height);
    std::cout << "OpenCL device: " << demosaic.deviceName() << "\n";

    std::error_code directoryError;
    fs::create_directories(options.outputDir, directoryError);
    if (directoryError)
    {
        std::cerr << "Error: cannot create " << options.outputDir << ": " << directoryError.message() << "\n";
        return EXIT_FAILURE;
    }

    camera.StartAcquisition();
    xiAPIplus_Image image;
    std::vector<std::uint8_t> rgb;
    double totalKernelMs = 0.0;
    double totalMs = 0.0;
    for (int i = 0; i < options.frameCount; ++i)
    {
        camera.GetNextImage(&image);
        const int rawStride = image.GetWidth() + image.GetPadding_X();
        const sample::DemosaicTiming timing = demosaic.process(image.GetPixels(), rawStride, redOffset->column,
                                                               redOffset->row, gains, rgb);
        totalKernelMs += timing.kernelMs;
        totalMs += timing.totalMs;

        char fileName[32];
        std::snprintf(fileName, sizeof(fileName), "image_%02d.tiff", i);
        const fs::path filePath = options.outputDir / fileName;
        sample::writeRgb8Tiff(filePath.string(), rgb.data(), static_cast<std::uint32_t>(width),
                              static_cast<std::uint32_t>(height));

        char line[160];
        std::snprintf(line, sizeof(line), "Frame %d/%d: kernel %.3f ms, upload + kernel + download %.3f ms, saved ",
                      i + 1, options.frameCount, timing.kernelMs, timing.totalMs);
        std::cout << line << filePath.string() << "\n";
    }

    char summary[128];
    std::snprintf(summary, sizeof(summary), "Average: kernel %.3f ms, upload + kernel + download %.3f ms",
                  totalKernelMs / options.frameCount, totalMs / options.frameCount);
    std::cout << summary << "\nDone\n";
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
