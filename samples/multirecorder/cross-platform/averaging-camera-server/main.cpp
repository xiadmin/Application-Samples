/*
 * Sample name: Averaging-camera-server
 * Category: Client-server / camera-side processing
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: xiAPI + MultiRecorder
 * Short description: Runs a MultiRecorder camera server with a user image
 * processor that averages every N frames on the camera computer, so that only
 * the averaged images travel to the client.
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <CameraServer.h>
#include <xiApi.h>

#include "averaging_processor.h"

struct Options
{
    std::string address = "localhost";
    int port = 9001;
};

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--address <host>] [--port <port>]\n"
              << "  --address  Address to listen at. Use 0.0.0.0 to accept clients from other\n"
              << "             computers. Default: localhost.\n"
              << "  --port     TCP port to listen at. Default: 9001.\n";
}

static bool parsePort(const std::string& text, int& value)
{
    try
    {
        std::size_t parsedLength = 0;
        value = std::stoi(text, &parsedLength);
        return parsedLength == text.size() && value > 0 && value <= 65535;
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
        if (argument == "--address")
            options.address = value;
        else if (argument == "--port")
            isValid = parsePort(value, options.port);
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

static int run(const Options& options)
{
    DWORD cameraCount = 0;
    const XI_RETURN status = xiGetNumberDevices(&cameraCount);
    if (status != XI_OK)
    {
        std::cerr << "Error: xiGetNumberDevices returned xiAPI error " << status << ", so no camera can be served\n";
        return EXIT_FAILURE;
    }
    if (cameraCount == 0)
    {
        std::cerr << "Error: no XIMEA cameras detected\n";
        return EXIT_FAILURE;
    }

    CxCameraServer server;
    // The server takes ownership of the processor and deletes it.
    server.setUserImageProcessor(std::make_unique<sample::AveragingProcessor>().release());

    std::cout << "Starting camera server at " << options.address << ":" << options.port
              << "; press Ctrl+C to stop" << std::endl;
    // Blocks while the server runs. The camera is opened when a client asks
    // for it, and every image request goes through the AveragingProcessor.
    if (!server.startAndListenAt(options.address, options.port, false, false))
    {
        std::cerr << "Error: could not start the camera server at " << options.address << ":" << options.port
                  << "; check that the address belongs to this computer and the port is free\n";
        return EXIT_FAILURE;
    }
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
