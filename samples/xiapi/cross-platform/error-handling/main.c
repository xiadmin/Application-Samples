/*
 * Sample name: Error-handling
 * Category: Error handling
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Shows how to detect and recover from an xiAPI error such
 * as XI_TIMEOUT.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xiApi.h>

static const int exposureUs = 100000;
/* Far shorter than the exposure, so the first frame cannot arrive in time. */
static const int shortTimeoutMs = 1;
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
    int exposure = 0;
    XI_IMG image;
    XI_RETURN stat;

    /* 1. Every xiAPI call returns an XI_RETURN code. A call on a handle that
     *    was never opened fails with XI_INVALID_HANDLE; no camera is needed. */
    stat = xiGetParamInt(NULL, XI_PRM_EXPOSURE, &exposure);
    if (stat == XI_INVALID_HANDLE)
        printf("Caught XI_INVALID_HANDLE (%d) from a call without an open camera\n", (int)stat);
    else
        printf("A call without an open camera returned %d\n", (int)stat);

    CE(xiGetNumberDevices(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    CE(xiStartAcquisition(cam));

    /* 2. Some errors are recoverable. Check XI_TIMEOUT explicitly instead of
     *    treating it as fatal, then retry with a timeout that fits the exposure. */
    stat = xiGetImage(cam, shortTimeoutMs, &image);
    if (stat == XI_TIMEOUT)
    {
        printf("Caught XI_TIMEOUT (%d) with a %d ms timeout; retrying with %d ms\n",
               (int)stat, shortTimeoutMs, grabTimeoutMs);
        CE(xiGetImage(cam, grabTimeoutMs, &image));
    }
    else if (stat != XI_OK)
    {
        fprintf(stderr, "Error: xiGetImage returned %d\n", (int)stat);
        return EXIT_FAILURE;
    }
    else
    {
        printf("A frame arrived within %d ms; no timeout this time\n", shortTimeoutMs);
    }

    printf("Got frame %ux%u nframe=%u\n", (unsigned)image.width, (unsigned)image.height,
           (unsigned)image.nframe);

    CE(xiStopAcquisition(cam));
    CE(xiCloseDevice(cam));

    printf("Done\n");
    return EXIT_SUCCESS;
}
