/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images and prints basic frame information.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xiApi.h>

static const int frameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;

/* Check error macro. Executes the call, reports and leaves main if it fails.
 * The camera handle is released by the driver when the process exits, so the
 * error path does not unwind explicitly. */
#define CE(func)                                                              \
    do                                                                        \
    {                                                                         \
        XI_RETURN stat = (func);                                              \
        if (stat != XI_OK)                                                    \
        {                                                                     \
            fprintf(stderr, "Error: %s returned %d\n", #func, (int)stat);     \
            return EXIT_FAILURE;                                              \
        }                                                                     \
    } while (0)

int main(void)
{
    HANDLE cam = NULL;
    XI_IMG image;

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiStartAcquisition(cam));

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    for (int i = 0; i < frameCount; i++)
    {
        CE(xiGetImage(cam, grabTimeoutMs, &image));

        printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n", i + 1, frameCount,
               (unsigned)image.width, (unsigned)image.height, (unsigned)image.nframe,
               image.bp ? (int)((const unsigned char *)image.bp)[0] : -1);
    }

    CE(xiStopAcquisition(cam));
    CE(xiCloseDevice(cam));

    printf("Done\n");
    return EXIT_SUCCESS;
}
