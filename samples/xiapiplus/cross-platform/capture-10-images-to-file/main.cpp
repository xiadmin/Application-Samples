/*
 * Sample name: Capture-10-images-to-file
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 images and saves them as TIFF files using
 * libtiff.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <xiapiplus.h>
#include <xiAPIplus_tiff.h>

static constexpr int frameCount = 10;
static constexpr int autoExposureFrameCount = 10;
static constexpr float exposureUs = 10000.0f;
static constexpr int grabTimeoutMs = 5000;

// PrintError() writes to stdout; samples report errors on stderr.
static void printXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::cerr << "Error " << exception.GetErrorNumber() << ": " << description << "\n";
}

int main()
{
    // xiAPIplus_Camera closes the camera in its destructor, so an exception
    // thrown below needs no explicit cleanup.
    xiAPIplus_Camera cam;

    try
    {
        if (cam.GetNumberOfConnectedCameras() == 0)
        {
            std::cerr << "Error: no XIMEA cameras detected\n";
            return EXIT_FAILURE;
        }

        std::cout << "Opening camera index 0\n";
        cam.OpenByID(0);
        cam.SetExposureTime(exposureUs);
        cam.SetBufferPolicy(XI_BP_SAFE);
        cam.EnableAutoExposureAutoGain();

        if (cam.IsSensorColor())
        {
            cam.SetImageDataFormat(XI_RGB24);
            cam.EnableWhiteBalanceAuto();
            cam.SetAutoExposureAutoGainExposurePriority(1.0f);
        }
        else
        {
            cam.SetImageDataFormat(XI_MONO8);
        }

        cam.SetNextImageTimeout_ms(grabTimeoutMs);
        cam.StartAcquisition();

        xiAPIplus_Image image;

        // Discard the first frames while auto exposure and white balance settle.
        for (int i = 0; i < autoExposureFrameCount; ++i)
            cam.GetNextImage(&image);

        for (int i = 0; i < frameCount; ++i)
        {
            char filename[32];

            cam.GetNextImage(&image);
            std::snprintf(filename, sizeof(filename), "image%03d.tif", i);
            // WriteImage() comes with xiAPIplus; it swaps BGR to RGB and drops
            // line padding, and prints the file it writes.
            WriteImage(&image, filename);
            std::cout << "Saved frame " << i + 1 << "/" << frameCount << " to " << filename << "\n";
        }

        cam.StopAcquisition();
        cam.Close();
    }
    catch (xiAPIplus_Exception& exception)
    {
        printXiError(exception);
        return EXIT_FAILURE;
    }
    catch (const char* p_message)
    {
        // WriteImage() reports libtiff failures by throwing a string literal.
        std::cerr << "Error: " << p_message << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
