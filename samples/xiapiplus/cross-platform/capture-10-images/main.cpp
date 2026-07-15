/* capture_images - XIMEA xiAPIplus capture sample (C++17)
 *
 * Opens the first available XIMEA camera, sets exposure to 100 ms,
 * captures 10 frames, prints per-frame metadata, then closes.
 *
 * Build: see README.md or scripts/build.py at the repo root.
 */

#include <cstdlib>
#include <iostream>

#include <xiApiPlus.h>

static constexpr int   frameCount    = 10;
static constexpr float exposureUs    = 100000.0f; 
static constexpr int   grabTimeoutMs = 5000;       

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
    catch (xiAPIplus_Exception exception)
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
        try { cam.StopAcquisition(); } catch (...) {}
    }
    if (cameraOpened)
    {
        try { cam.Close(); } catch (...) {}
    }

    return EXIT_FAILURE;
}
