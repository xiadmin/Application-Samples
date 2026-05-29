/* capture_images - XIMEA xiAPIplus capture sample (C++17)
 *
 * Opens the first available XIMEA camera, sets exposure to 100 ms,
 * captures 10 frames, prints per-frame metadata, then closes.
 *
 * Build: see CMakeLists.txt or build.ps1 at the repo root.
 */

#include <xiApiPlus.h>
#include <cstdlib>
#include <iostream>

static constexpr int   frameCount    = 10;
static constexpr float exposureUs    = 100000.0f; // 100 ms in microseconds
static constexpr int   grabTimeoutMs = 5000;       // must exceed exposure

static int runCapture(xiAPIplus_Camera& cam)
{
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
    std::cout << "Done\n";
    return EXIT_SUCCESS;
}

int main()
{
    xiAPIplus_Camera cam;
    unsigned long count = cam.GetNumberOfConnectedCameras();

    if (count == 0)
    {
        std::cerr << "Error: no XIMEA cameras detected\n";
        return EXIT_FAILURE;
    }

    std::cout << "Found " << count << " camera(s), opening index 0\n";

    cam.OpenByID(0);
    int ret = runCapture(cam);
    cam.Close();
    return ret;
}
