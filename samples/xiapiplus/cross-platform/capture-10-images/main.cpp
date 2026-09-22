/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 images.
 */

#include <cstdlib>
#include <iostream>

#include <xiapiplus.h>

static constexpr int frameCount = 10;
static constexpr float exposureUs = 100000.0f;
static constexpr int grabTimeoutMs = 5000;

int main()
{
    // xiAPIplus_Camera closes the camera in its destructor, so an exception
    // thrown below needs no explicit cleanup.
    xiAPIplus_Camera cam;

    try
    {
        unsigned long count = cam.GetNumberOfConnectedCameras();

        if (count == 0)
        {
            std::cerr << "Error: no XIMEA cameras detected\n";
            return EXIT_FAILURE;
        }

        std::cout << "Found " << count << " camera(s), opening index 0\n";
        cam.OpenByID(0);

        cam.SetExposureTime(exposureUs);
        std::cout << "Exposure: " << static_cast<int>(exposureUs) << " us ("
                  << static_cast<int>(exposureUs) / 1000 << " ms)\n";

        cam.SetNextImageTimeout_ms(grabTimeoutMs);
        cam.StartAcquisition();
        std::cout << "Capturing " << frameCount << " frames\n";

        xiAPIplus_Image img;
        for (int i = 0; i < frameCount; ++i)
        {
            cam.GetNextImage(&img);
            unsigned char* data = img.GetPixels();
            int firstByte = data ? static_cast<int>(data[0]) : -1;
            std::cout << "Frame " << i + 1 << "/" << frameCount
                      << ": " << img.GetWidth() << "x" << img.GetHeight()
                      << " nframe=" << img.GetFrameNumber()
                      << " first_byte=" << firstByte << "\n";
        }

        cam.StopAcquisition();
        cam.Close();
    }
    catch (xiAPIplus_Exception& exception)
    {
        exception.PrintError();
        return EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
