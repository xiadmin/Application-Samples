/* capture_images - XIMEA xiAPI capture sample (C11, no parameters)
 *
 * Opens the first available XIMEA camera, sets exposure to 100 ms,
 * captures 10 frames, prints per-frame metadata, then closes.
 *
 * Build: see CMakeLists.txt or build.sh / build.bat at the repo root.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xiApi.h>

static const int frameCount = 10;
static const int exposureUs = 100000;  /* 100 ms in microseconds */
static const int grabTimeoutMs = 5000; /* must exceed exposure */

/* Print error and return 0 on failure, 1 on success. */
static int xiOk(XI_RETURN st, const char *where)
{
    if (st == XI_OK)
        return 1;
    fprintf(stderr, "Error: %s returned %d\n", where, (int)st);
    return 0;
}

static int runCapture(HANDLE cam)
{
    XI_IMG img;
    int i;

    if (!xiOk(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs),
              "xiSetParamInt(exposure)"))
        return EXIT_FAILURE;

    printf("Exposure: %d us (%d ms)\n", exposureUs, exposureUs / 1000);

    if (!xiOk(xiStartAcquisition(cam), "xiStartAcquisition"))
        return EXIT_FAILURE;

    printf("Capturing %d frames\n", frameCount);

    for (i = 0; i < frameCount; i++)
    {
        memset(&img, 0, sizeof(img));
        img.size = sizeof(img);

        if (!xiOk(xiGetImage(cam, grabTimeoutMs, &img), "xiGetImage"))
        {
            fprintf(stderr, "Error: failed on frame %d/%d\n", i + 1, frameCount);
            xiStopAcquisition(cam);
            return EXIT_FAILURE;
        }

        printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n",
               i + 1, frameCount,
               (unsigned)img.width, (unsigned)img.height,
               (unsigned)img.nframe,
               img.bp ? (int)((unsigned char *)img.bp)[0] : -1);
    }

    xiStopAcquisition(cam);
    printf("Done\n");
    return EXIT_SUCCESS;
}

int main(void)
{
    uint32_t count = 0;
    HANDLE cam = NULL;
    int ret;

    if (!xiOk(xiGetNumberDevices(&count), "xiGetNumberDevices"))
        return EXIT_FAILURE;

    if (count == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Found %u camera(s), opening index 0\n", (unsigned)count);

    if (!xiOk(xiOpenDevice(0, &cam), "xiOpenDevice"))
        return EXIT_FAILURE;

    ret = runCapture(cam);
    xiCloseDevice(cam);
    return ret;
}