/*
 * Sample name: Capture-10-images-to-dng
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 raw images and saves them as DNG files using
 * the XIMEA DNG Store library.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <xiapiplus.h>
#include <xiapi_dng_store.h>

static constexpr int frameCount = 10;
static constexpr float exposureUs = 10000.0f;
static constexpr int grabTimeoutMs = 5000;

// PrintError() writes to stdout; samples report errors on stderr.
static void printXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::cerr << "Error " << exception.GetErrorNumber() << ": " << description << "\n";
}

// The DNG Store library is a plain C API; turn its XI_RETURN codes into the
// same exception type xiAPIplus throws, so one handler covers both.
static void checkDng(XI_RETURN result, const char* p_location)
{
    if (result != XI_OK)
        ReportException(result, p_location);
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
        // DNG stores the sensor data as captured: Bayer data on a color camera.
        cam.SetImageDataFormat(XI_RAW8);
        cam.SetExposureTime(exposureUs);
        cam.SetNextImageTimeout_ms(grabTimeoutMs);
        cam.StartAcquisition();

        // The DNG Store calls take the raw xiAPI handle and XI_IMG, which
        // xiAPIplus exposes. Metadata is read once; settings do not change.
        XI_DNG_METADATA metadata;
        checkDng(xidngFillMetadataFromCameraParams(cam.GetCameraHandle(), &metadata),
                 "xidngFillMetadataFromCameraParams");

        xiAPIplus_Image image;
        for (int i = 0; i < frameCount; ++i)
        {
            char filename[32];

            cam.GetNextImage(&image);
            std::snprintf(filename, sizeof(filename), "image%03d.dng", i);
            checkDng(xidngStore(filename, image.GetXI_IMG(), &metadata), "xidngStore");

            std::cout << "Saved frame " << i + 1 << "/" << frameCount << " (" << image.GetWidth()
                      << "x" << image.GetHeight() << ") to " << filename << "\n";
        }

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
