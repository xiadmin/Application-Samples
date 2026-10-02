/*
 * Sample name: Error-handling
 * Category: Error handling
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Shows how to detect and recover from an xiAPI error such
 * as XI_TIMEOUT.
 */

#include <cstdlib>
#include <iostream>

#include <xiapiplus.h>

static constexpr float exposureUs = 100000.0f;
// Far shorter than the exposure, so the first frame cannot arrive in time.
static constexpr int shortTimeoutMs = 1;
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
    // 1. xiAPIplus reports every failed call as an xiAPIplus_Exception that
    //    carries the XI_RETURN code. A camera object that was never opened
    //    fails with XI_INVALID_HANDLE; no camera is needed.
    try
    {
        xiAPIplus_Camera unopened;
        unopened.GetExposureTime();
        std::cout << "A call without an open camera succeeded unexpectedly\n";
    }
    catch (xiAPIplus_Exception& exception)
    {
        if (exception.GetErrorNumber() == XI_INVALID_HANDLE)
            std::cout << "Caught XI_INVALID_HANDLE (" << XI_INVALID_HANDLE
                      << ") from a call without an open camera\n";
        else
            std::cout << "A call without an open camera failed with " << exception.GetErrorNumber() << "\n";
    }

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
        cam.StartAcquisition();

        // 2. Some errors are recoverable. Catch XI_TIMEOUT explicitly, retry
        //    with a timeout that fits the exposure, and rethrow anything else.
        xiAPIplus_Image image;
        try
        {
            cam.SetNextImageTimeout_ms(shortTimeoutMs);
            cam.GetNextImage(&image);
            std::cout << "A frame arrived within " << shortTimeoutMs << " ms; no timeout this time\n";
        }
        catch (xiAPIplus_Exception& exception)
        {
            if (exception.GetErrorNumber() != XI_TIMEOUT)
                throw;

            std::cout << "Caught XI_TIMEOUT (" << XI_TIMEOUT << ") with a " << shortTimeoutMs
                      << " ms timeout; retrying with " << grabTimeoutMs << " ms\n";
            cam.SetNextImageTimeout_ms(grabTimeoutMs);
            cam.GetNextImage(&image);
        }

        std::cout << "Got frame " << image.GetWidth() << "x" << image.GetHeight()
                  << " nframe=" << image.GetFrameNumber() << "\n";

        cam.StopAcquisition();
        cam.Close();
    }
    catch (xiAPIplus_Exception& exception)
    {
        printXiError(exception);
        return EXIT_FAILURE;
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
