/*
 * Sample name: Offline-processing
 * Category: Image processing
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPIplus
 * Short description: Captures 10 RAW8 images, converts them to RGB24 with
 * xiAPI offline processing after acquisition, and saves them as TIFF files.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <vector>

#include <xiapiplus.h>
#include <xiAPIplus_tiff.h>

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

int main()
{
    std::vector<std::vector<unsigned char>> frames;
    std::vector<char> context;
    int outputFormat = XI_MONO8;

    try
    {
        {
            // The camera is scoped to this block, so it is closed before the
            // stored frames are processed below.
            xiAPIplus_Camera cam;

            if (cam.GetNumberOfConnectedCameras() == 0)
            {
                std::cerr << "Error: no XIMEA cameras detected\n";
                return EXIT_FAILURE;
            }

            std::cout << "Opening camera index 0\n";
            cam.OpenByID(0);
            cam.SetImageDataFormat(XI_RAW8);
            cam.SetExposureTime(exposureUs);
            cam.SetNextImageTimeout_ms(grabTimeoutMs);

            if (cam.IsSensorColor())
            {
                // Not applied to the RAW8 frames; recorded in the context
                // list below and applied by the offline processing.
                cam.SetWhiteBalanceRed(1.05f);
                cam.SetWhiteBalanceGreen(1.0f);
                cam.SetWhiteBalanceBlue(2.10f);
                outputFormat = XI_RGB24;
            }

            cam.StartAcquisition();

            xiAPIplus_Image image;
            for (int i = 0; i < frameCount; ++i)
            {
                cam.GetNextImage(&image);
                const unsigned char* p_pixels = image.GetPixels();
                frames.emplace_back(p_pixels, p_pixels + image.GetPixelsArraySize());
            }

            cam.StopAcquisition();

            // The context list holds every setting the frames were captured
            // with. It is all the processing needs, so the camera can close.
            char sizeQuery[] = XI_PRM_API_CONTEXT_LIST XI_PRMM_REQ_VAL_BUFFER_SIZE;
            int contextSize = 0;
            cam.GetXIAPIParamInt(sizeQuery, &contextSize);
            context.resize(static_cast<size_t>(contextSize) + 1, '\0');
            cam.GetApiContextList(context.data(), static_cast<int>(context.size()));
            cam.Close();
        }

        std::cout << "Camera closed; processing " << frameCount << " stored frames offline\n";

        // xiAPIplus_ImageProcessing opens the processing in its constructor
        // and closes it in its destructor.
        xiAPIplus_ImageProcessing proc;
        proc.SetParam(XI_PRM_API_CONTEXT_LIST, context.data(),
                      static_cast<DWORD>(std::strlen(context.data())), xiTypeString);
        proc.SetParam(XI_PRM_IMAGE_DATA_FORMAT, &outputFormat, sizeof(outputFormat), xiTypeInteger);

        for (int i = 0; i < frameCount; ++i)
        {
            char filename[32];

            proc.PushImage(frames[i].data());
            proc.PullImage();

            // GetImage() hands over a malloc'd copy without marking it as one;
            // SetCopy(true) makes the xiAPIplus_Image destructor free it.
            xiAPIplus_Image processed;
            proc.GetImage(&processed);
            processed.SetCopy(true);

            std::snprintf(filename, sizeof(filename), "image%03d.tif", i);
            WriteImage(&processed, filename);
            std::cout << "Processed frame " << i + 1 << "/" << frameCount << " (" << processed.GetWidth()
                      << "x" << processed.GetHeight() << ") to " << filename << "\n";
        }
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
    catch (const std::bad_alloc&)
    {
        std::cerr << "Error: could not allocate RAM for " << frameCount << " images\n";
        return EXIT_FAILURE;
    }

    std::cout << "Done\n";
    return EXIT_SUCCESS;
}
