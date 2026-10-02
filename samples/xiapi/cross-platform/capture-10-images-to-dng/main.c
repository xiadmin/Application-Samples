/*
 * Sample name: Capture-10-images-to-dng
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 raw images and saves them as DNG files using
 * the XIMEA DNG Store library.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xiApi.h>
#include <xiapi_dng_store.h>

static const int frameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;

/* Check error macro. Executes the call, reports and leaves main if it fails.
 * The camera handle is released by the driver when the process exits, so the
 * error path does not unwind explicitly. */
#define CE(func)                                                            \
    do                                                                      \
    {                                                                       \
        XI_RETURN stat = (func);                                            \
        if (stat != XI_OK)                                                  \
        {                                                                   \
            fprintf(stderr, "Error: %s returned %d\n", #func, (int)stat);   \
            return EXIT_FAILURE;                                            \
        }                                                                   \
    } while (0)

int main(void)
{
    HANDLE cam = NULL;
    DWORD deviceCount = 0;
    XI_IMG image;
    XI_DNG_METADATA metadata;

    CE(xiGetNumberDevices(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    /* DNG stores the sensor data as captured: Bayer data on a color camera. */
    CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    CE(xiStartAcquisition(cam));

    /* Reads model, serial number, CFA, white balance and the like from the
     * camera once; the settings do not change between frames here. */
    CE(xidngFillMetadataFromCameraParams(cam, &metadata));

    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];

        CE(xiGetImage(cam, grabTimeoutMs, &image));

        snprintf(filename, sizeof(filename), "image%03d.dng", i);
        CE(xidngStore(filename, &image, &metadata));

        printf("Saved frame %d/%d (%ux%u) to %s\n", i + 1, frameCount,
               (unsigned)image.width, (unsigned)image.height, filename);
    }

    CE(xiStopAcquisition(cam));
    CE(xiCloseDevice(cam));

    printf("Done\n");
    return EXIT_SUCCESS;
}
