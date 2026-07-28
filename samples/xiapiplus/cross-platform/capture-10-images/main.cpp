/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 images.
 *
 * Copyright (c) 2026 XIMEA s.r.o.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <cstdlib>
#include <iostream>

#include <xiapiplus.h>

static constexpr int frameCount = 10;
static constexpr float exposureUs = 100000.0f;
static constexpr int grabTimeoutMs = 5000;

int main()
{
    xiAPIplus_Camera cam;
    bool acquisitionStarted = false;
    bool cameraOpened = false;

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
        cameraOpened = true;

        cam.SetExposureTime(exposureUs);
        std::cout << "Exposure: " << static_cast<int>(exposureUs) << " us ("
                  << static_cast<int>(exposureUs) / 1000 << " ms)\n";

        cam.SetNextImageTimeout_ms(grabTimeoutMs);
        cam.StartAcquisition();
        acquisitionStarted = true;
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
        acquisitionStarted = false;
        std::cout << "Done\n";

        cam.Close();
        cameraOpened = false;
        return EXIT_SUCCESS;
    }
    catch (xiAPIplus_Exception& exception)
    {
        exception.PrintError();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
    }
    catch (...)
    {
        std::cerr << "Error: unknown exception occurred\n";
    }

    if (acquisitionStarted)
    {
        try
        {
            cam.StopAcquisition();
        }
        catch (...)
        {
        }
    }
    if (cameraOpened)
    {
        try
        {
            cam.Close();
        }
        catch (...)
        {
        }
    }

    return EXIT_FAILURE;
}
