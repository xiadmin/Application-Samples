
/* capture_images - XIMEA xiAPI capture sample (C)
 *
 * Opens the first available XIMEA camera, sets exposure to 100 ms,
 * captures 10 frames, prints per-frame metadata, then closes.
 *
 * Build: see CMakeLists.txt or build.ps1 at the repo root.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xiApi.h>

const int frameCount = 10;
const int exposureUs = 100000;
const int grabTimeoutMs = 5000;

#define CE(func)                                       \
    do                                                 \
    {                                                  \
        XI_RETURN stat = (func);                       \
        if (stat != XI_OK)                             \
        {                                              \
            fprintf(stderr, "Error: %s returned %d\n", \
                    #func, (int)stat);                 \
            goto cleanup;                              \
        }                                              \
    } while (0)

int main(void)
{
    uint32_t count = 0;
    HANDLE cam = NULL;
    XI_IMG img;
    XI_RETURN st;
    int ret = EXIT_SUCCESS;

    CE(xiGetNumberDevices(&count));
    if (count == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Found %u camera(s), opening index 0\n", (unsigned)count);

    CE(xiOpenDevice(0, &cam));

    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));

    printf("Exposure: %d us (%d ms)\n", exposureUs, exposureUs / 1000);

    CE(xiStartAcquisition(cam));

    printf("Capturing %d frames\n", frameCount);

    for (int i = 0; i < frameCount; i++)
    {
        memset(&img, 0, sizeof(img));
        img.size = sizeof(img);

        CE(xiGetImage(cam, grabTimeoutMs, &img));

        printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n", i + 1, frameCount,
               (unsigned)img.width, (unsigned)img.height, (unsigned)img.nframe,
               img.bp ? (int)((unsigned char *)img.bp)[0] : -1);
    }

cleanup:
    if (cam)
    {
        xiStopAcquisition(cam);
        xiCloseDevice(cam);
    }

    if (ret == EXIT_SUCCESS)
        printf("Done\n");

    return ret;
}
