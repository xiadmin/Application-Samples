/*
 * Sample name: Capture-50-images-to-video-file-opencv
 * Category: Video recording / OpenCV integration
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 50 grayscale images with the xiAPIplus OpenCV
 * wrapper, shows them in a window and records them to a Motion JPEG AVI file.
 */

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <xiApiPlusOcv.hpp>

static constexpr int frameCount = 50;
static constexpr float exposureUs = 40000.0f;
static constexpr double videoFps = 25.0;
static constexpr int displayDelayMs = 20;
static constexpr const char* windowName = "XIMEA camera";

struct Options
{
    std::string outputPath = "capture.avi";
};

// xiApiPlusOcv.hpp carries its own copy of xiAPIplus_Exception, whose
// PrintError() writes to stdout; samples report errors on stderr.
static void printXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::cerr << "Error " << exception.GetErrorNumber() << ": " << description << "\n";
}

static void printUsage(const char* p_program)
{
    std::cout << "Usage: " << p_program << " [--output <file.avi>]\n"
              << "  --output  Video file to write. Default: capture.avi in the current directory.\n";
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
        if (argument == "--output" && i + 1 < argc)
            options.outputPath = argv[++i];
        else
        {
            std::cerr << "Error: unknown or incomplete argument '" << argument << "'\n";
            return std::nullopt;
        }
    }
    return options;
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

    // xiAPIplusCameraOcv closes the camera in its destructor and
    // cv::VideoWriter finalizes the file in its destructor, so an exception
    // thrown below needs no explicit cleanup.
    xiAPIplusCameraOcv cam;

    try
    {
        if (cam.GetNumberOfConnectedCameras() == 0)
        {
            std::cerr << "Error: no XIMEA cameras detected\n";
            return EXIT_FAILURE;
        }

        std::cout << "Opening first camera\n";
        cam.OpenFirst();
        cam.SetExposureTime(exposureUs);
        // The video is recorded in grayscale; MONO8 works on color cameras too.
        cam.SetImageDataFormat(XI_MONO8);

        const cv::Size frameSize(cam.GetWidth(), cam.GetHeight());
        // Motion JPEG in AVI is encoded by OpenCV itself, so it works without
        // FFmpeg or system codecs on every platform.
        cv::VideoWriter video(options->outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFps, frameSize,
                              false);
        if (!video.isOpened())
        {
            std::cerr << "Error: cannot open " << options->outputPath << " for writing\n";
            return EXIT_FAILURE;
        }

        std::cout << "Starting acquisition\n";
        cam.StartAcquisition();

        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        for (int i = 0; i < frameCount; ++i)
        {
            const cv::Mat image = cam.GetNextImageOcvMat();
            video.write(image);
            cv::imshow(windowName, image);
            cv::waitKey(displayDelayMs);
            std::cout << "Recorded frame " << i + 1 << "/" << frameCount << "\n";
        }

        cam.StopAcquisition();
        cam.Close();
        video.release();
        cv::destroyAllWindows();
    }
    catch (xiAPIplus_Exception& exception)
    {
        printXiError(exception);
        return EXIT_FAILURE;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr << "OpenCV error: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Saved " << options->outputPath << "\nDone\n";
    return EXIT_SUCCESS;
}
