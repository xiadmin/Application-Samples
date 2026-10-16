/*
 * Sample name: Averaging-camera-client
 * Category: Client-server / camera-side processing
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: xiAPI + MultiRecorder
 * Short description: Connects to an averaging camera server, configures its
 * camera, asks the server to average N frames per image and prints the size,
 * frame number and mean value of every averaged image it receives.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinyxml.h>

#include <RecorderApi.h>
#include <RemoteCameraClient.h>
#include <xiApi.h>

#include "remote_camera_session.h"

static constexpr unsigned frameTimeoutMs = 5000;
// Content type of a response that carries an XI_IMG header and its pixels.
static const std::string imageContentType = "application/octet-stream";

struct Options
{
    std::string host = "localhost";
    int port = 9001;
    int averagingCount = 4;
    int frameCount = 100;
    int exposureUs = 10000;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program
              << " [--server <host:port>] [--averaging <n>] [--frame-count <n>] [--exposure-us <us>]\n"
              << "  --server       Camera server to connect to. Default: localhost:9001.\n"
              << "  --averaging    Frames the server averages into one image. Default: 4.\n"
              << "  --frame-count  Camera frames to request; the client receives\n"
              << "                 frame-count / averaging images. Default: 100.\n"
              << "  --exposure-us  Exposure time in microseconds. Default: 10000.\n";
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

// Splits "host:port"; the port must be given.
static bool parseServer(const std::string& text, std::string& host, int& port)
{
    const std::size_t separator = text.rfind(':');
    if (separator == std::string::npos || separator == 0)
        return false;
    host = text.substr(0, separator);
    return parsePositiveInt(text.substr(separator + 1), port) && port <= 65535;
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
        if (argument == "--server")
            isValid = parseServer(value, options.host, options.port);
        else if (argument == "--averaging")
            isValid = parsePositiveInt(value, options.averagingCount);
        else if (argument == "--frame-count")
            isValid = parsePositiveInt(value, options.frameCount);
        else if (argument == "--exposure-us")
            isValid = parsePositiveInt(value, options.exposureUs);
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

// Mean of the image's pixel values, skipping the row padding.
static double meanPixelValue(const XI_IMG& image)
{
    const std::size_t rowBytes = static_cast<std::size_t>(image.width) * (image.frm == XI_RGB24 ? 3 : 1);
    const std::size_t stride = rowBytes + image.padding_x;
    const auto* p_pixels = static_cast<const std::uint8_t*>(image.bp);
    std::uint64_t sum = 0;
    for (std::size_t row = 0; row < image.height; ++row)
    {
        for (std::size_t column = 0; column < rowBytes; ++column)
            sum += p_pixels[row * stride + column];
    }
    const std::size_t count = rowBytes * image.height;
    return count > 0 ? static_cast<double>(sum) / count : 0.0;
}

static void configureCamera(CxRemoteCameraClient& camera, const Options& options)
{
    sample::checkXi(camera.setParamInt(XI_PRM_EXPOSURE, options.exposureUs), "setParamInt(XI_PRM_EXPOSURE)");

    // The server averages 8-bit data only: RGB24 on color cameras with
    // automatic white balance, MONO8 on monochrome ones.
    int isColor = 0;
    sample::checkXi(camera.getParamInt(XI_PRM_IMAGE_IS_COLOR, isColor), "getParamInt(XI_PRM_IMAGE_IS_COLOR)");
    sample::checkXi(camera.setParamInt(XI_PRM_IMAGE_DATA_FORMAT, isColor ? XI_RGB24 : XI_MONO8),
                    "setParamInt(XI_PRM_IMAGE_DATA_FORMAT)");
    if (isColor)
        sample::checkXi(camera.setParamInt(XI_PRM_AUTO_WB, 1), "setParamInt(XI_PRM_AUTO_WB)");

    // A user image processing request reaches the server's AveragingProcessor.
    TiXmlElement request(XI_SRV_TAG_REQUEST);
    request.SetAttribute("averaging", options.averagingCount);
    std::string response;
    bool isAccepted = camera.userImageProcessorCommand(&request, &response);
    std::string details;
    if (isAccepted)
        xiRecorderApi::parseXmlResponse(response, isAccepted, details);
    if (!isAccepted)
        throw std::runtime_error("the camera server rejected the averaging request: " + details);
    std::cout << "Server averages " << options.averagingCount << " frames per image\n";
}

static int run(const Options& options)
{
    const std::string address = options.host + ":" + std::to_string(options.port);
    std::string connectError;
    const std::list<CxRemoteCameraClient::SxRemoteCameraDesc> cameras =
        CxRemoteCameraClient::listCamerasAtAddress(options.host, options.port, connectError);
    if (cameras.empty())
    {
        if (!connectError.empty())
        {
            std::cerr << "Error: could not connect to the camera server at " << address << " (" << connectError
                      << "); start averaging-camera-server first\n";
        }
        else
            std::cerr << "Error: the camera server at " << address << " reports no camera\n";
        return EXIT_FAILURE;
    }

    const CxRemoteCameraClient::SxRemoteCameraDesc& description = cameras.front();
    std::cout << "Opening camera " << description.m_sName << " (" << description.m_sSerial << ") at " << address
              << "\n";
    CxRemoteCameraClient* p_camera = CxRemoteCameraClient::openCamera(
        description.m_sIPAddress, description.m_iPort, description.m_sName, description.m_sSerial, true);
    if (p_camera == nullptr)
    {
        std::cerr << "Error: could not open the camera on the camera server at " << address << "\n";
        return EXIT_FAILURE;
    }
    sample::RemoteCameraSession session(p_camera);
    CxRemoteCameraClient& camera = session.camera();
    configureCamera(camera, options);

    int payloadSize = 0;
    sample::checkXi(camera.getParamInt(XI_PRM_IMAGE_PAYLOAD_SIZE, payloadSize), "getParamInt(XI_PRM_IMAGE_PAYLOAD_SIZE)");
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(payloadSize));

    session.startAcquisition();
    int imageCount = 0;
    for (int i = 0; i < options.frameCount; ++i)
    {
        std::string response;
        std::string contentType;
        if (!camera.getImageForResponse(frameTimeoutMs, response, contentType))
        {
            std::cerr << "Error: the camera server did not answer the request for frame " << i + 1 << "\n";
            return EXIT_FAILURE;
        }

        if (contentType != imageContentType)
        {
            // Text answers, such as "Averaging frame 2/4", arrive wrapped in a
            // <response> element.
            bool isSuccess = false;
            std::string details;
            xiRecorderApi::parseXmlResponse(response, isSuccess, details);
            if (!isSuccess)
            {
                std::cerr << "Error: the camera server could not process frame " << i + 1 << ": " << details << "\n";
                return EXIT_FAILURE;
            }
            std::cout << "Server: " << details << "\n";
            continue;
        }

        // The response holds the XI_IMG header and then the pixels, so it is
        // always at least as large as the pixel data.
        if (pixels.size() < response.size())
            pixels.resize(response.size());
        XI_IMG image{};
        image.size = sizeof(XI_IMG);
        image.bp = pixels.data();
        image.bp_size = static_cast<DWORD>(pixels.size());
        sample::checkXi(CxRemoteCameraClient::fillXiImgFromResponse(&image, response), "fillXiImgFromResponse");

        ++imageCount;
        char line[128];
        std::snprintf(line, sizeof(line), "Averaged image %d: %ux%u, last frame number %u, mean value %.1f", imageCount,
                      static_cast<unsigned>(image.width), static_cast<unsigned>(image.height),
                      static_cast<unsigned>(image.nframe), meanPixelValue(image));
        std::cout << line << "\n";
    }
    session.stopAcquisition();

    const int expectedCount = options.frameCount / options.averagingCount;
    if (imageCount != expectedCount)
    {
        std::cerr << "Error: received " << imageCount << " of " << expectedCount << " averaged images\n";
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
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
