/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xiApi.h>

const int frameCount = 10;
const int exposureUs = 10000;
const int grabTimeoutMs = 5000;

static int cleanupCamera(HANDLE cam, int ret)
{
    if (cam)
    {
        xiStopAcquisition(cam);
        xiCloseDevice(cam);
    }

    if (ret == EXIT_SUCCESS)
        printf("Done\n");

    return ret;
}

#define CE(func)                                       \
    do                                                 \
    {                                                  \
        XI_RETURN stat = (func);                       \
        if (stat != XI_OK)                             \
        {                                              \
            fprintf(stderr, "Error: %s returned %d\n", \
                    #func, (int)stat);                 \
            return cleanupCamera(cam, EXIT_FAILURE);   \
        }                                              \
    } while (0)

int main(void)
{
    uint32_t count = 0;
    HANDLE cam = NULL;
    XI_IMG img;

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

    return cleanupCamera(cam, EXIT_SUCCESS);
}
