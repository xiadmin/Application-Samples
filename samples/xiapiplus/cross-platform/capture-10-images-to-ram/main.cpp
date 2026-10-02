/*
 * Sample name: Capture-10-images-to-ram
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 images into application-owned RAM.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <utility>
#include <vector>

#include <xiapiplus.h>

static constexpr int frameCount = 10;
static constexpr float exposureUs = 10000.0f;
static constexpr int grabTimeoutMs = 5000;

// A frame copied out of the driver buffer, owned by the application.
struct StoredFrame
{
    int width = 0;
    int height = 0;
    int frameNumber = 0;
    std::vector<unsigned char> pixels;
};

// PrintError() writes to stdout; samples report errors on stderr.
static void printXiError(xiAPIplus_Exception& exception)
{
    char description[256] = "";
    exception.GetDescription(description, sizeof(description) - 1);
    std::cerr << "Error " << exception.GetErrorNumber() << ": " << description << "\n";
}

int main()
{
    std::vector<StoredFrame> frames;
    frames.reserve(frameCount);

    try
    {
        // The camera is scoped to this block, so it is closed before the
        // stored frames are read below.
        xiAPIplus_Camera cam;

        if (cam.GetNumberOfConnectedCameras() == 0)
        {
            std::cerr << "Error: no XIMEA cameras detected\n";
            return EXIT_FAILURE;
        }

        std::cout << "Opening camera index 0\n";
        cam.OpenByID(0);
        cam.SetImageDataFormat(XI_RAW8);
        cam.SetBufferPolicy(XI_BP_SAFE);
        cam.SetExposureTime(exposureUs);
        cam.SetNextImageTimeout_ms(grabTimeoutMs);
        cam.StartAcquisition();

        xiAPIplus_Image image;
        for (int i = 0; i < frameCount; ++i)
        {
            cam.GetNextImage(&image);

            // Copy into a std::vector rather than xiAPIplus_Image::CopyTo():
            // the copy that CopyTo() allocates is not freed by the destructor.
            StoredFrame frame;
            frame.width = image.GetWidth();
            frame.height = image.GetHeight();
            frame.frameNumber = image.GetFrameNumber();
            frame.pixels.resize(static_cast<size_t>(image.GetPixelsArraySize()));
            std::memcpy(frame.pixels.data(), image.GetPixels(), frame.pixels.size());
            frames.push_back(std::move(frame));
        }

        cam.StopAcquisition();
        cam.Close();
    }
    catch (xiAPIplus_Exception& exception)
    {
        printXiError(exception);
        return EXIT_FAILURE;
    }
    catch (const std::bad_alloc&)
    {
        std::cerr << "Error: could not allocate RAM for " << frameCount << " images\n";
        return EXIT_FAILURE;
    }

    std::cout << "Camera closed; reading retained images from RAM\n";
    for (size_t i = 0; i < frames.size(); ++i)
    {
        const StoredFrame& frame = frames[i];
        int firstByte = frame.pixels.empty() ? -1 : static_cast<int>(frame.pixels[0]);
        std::cout << "Frame " << i + 1 << "/" << frameCount << ": " << frame.width << "x"
                  << frame.height << " nframe=" << frame.frameNumber
                  << " first_byte=" << firstByte << "\n";
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
