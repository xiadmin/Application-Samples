/*
 * Sample name: Capture-50-images-and-display-opencv
 * Category: Display / OpenCV integration
 * OS platform: Cross-platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 50 images as OpenCV matrices with the xiAPIplus
 * OpenCV wrapper and shows them in a window.
 */

#include <cstdlib>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <xiApiPlusOcv.hpp>

static constexpr int frameCount = 50;
static constexpr float exposureUs = 10000.0f;
static constexpr int displayDelayMs = 20;
static constexpr const char* windowName = "XIMEA camera";

// xiApiPlusOcv.hpp carries its own copy of xiAPIplus_Exception, whose
// PrintError() writes to stdout; samples report errors on stderr.
static void printXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::cerr << "Error " << exception.GetErrorNumber() << ": " << description << "\n";
}

int main()
{
    // xiAPIplusCameraOcv closes the camera in its destructor, so an exception
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

        std::cout << "Starting acquisition\n";
        cam.StartAcquisition();
        const XI_IMG_FORMAT format = cam.GetImageDataFormat();
        const bool is16Bit = format == XI_RAW16 || format == XI_MONO16;

        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
        for (int i = 0; i < frameCount; ++i)
        {
            cv::Mat image = cam.GetNextImageOcvMat();
            unsigned firstPixel = 0;
            if (is16Bit)
            {
                // Sensors deliver 10 or 12 significant bits in a 16-bit
                // container; stretch them to the full range so the image is
                // not almost black on screen.
                cv::normalize(image, image, 0, 65535, cv::NORM_MINMAX);
                firstPixel = image.at<unsigned short>(0, 0);
            }
            else
            {
                firstPixel = image.at<unsigned char>(0, 0);
            }
            std::cout << "Frame " << i + 1 << "/" << frameCount << ": first pixel " << firstPixel << "\n";

            cv::imshow(windowName, image);
            cv::waitKey(displayDelayMs);
        }

        cam.StopAcquisition();
        cam.Close();
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

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
