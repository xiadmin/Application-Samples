/*
 * Sample name: Capture-10-images-to-file
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images and saves them as TIFF files.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xiApi.h>

#include "tiff_writer.h"

static const int frameCount = 10;
static const int autoExposureFrameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;

static int cleanupCamera(HANDLE cam, int isAcquisitionStarted, int result)
{
    XI_RETURN stat;

    if (isAcquisitionStarted)
    {
        stat = xiStopAcquisition(cam);
        if (stat != XI_OK)
        {
            fprintf(stderr, "Error: xiStopAcquisition returned %d\n", (int)stat);
            result = EXIT_FAILURE;
        }
    }

    if (cam)
    {
        stat = xiCloseDevice(cam);
        if (stat != XI_OK)
        {
            fprintf(stderr, "Error: xiCloseDevice returned %d\n", (int)stat);
            result = EXIT_FAILURE;
        }
    }

    if (result == EXIT_SUCCESS)
        printf("Done\n");

    return result;
}

#define CE(func)                                                              \
    do                                                                        \
    {                                                                         \
        XI_RETURN stat = (func);                                              \
        if (stat != XI_OK)                                                    \
        {                                                                     \
            fprintf(stderr, "Error: %s returned %d\n", #func, (int)stat);   \
            return cleanupCamera(cam, isAcquisitionStarted, EXIT_FAILURE);    \
        }                                                                     \
    } while (0)

int main(void)
{
    HANDLE cam = NULL;
    int isAcquisitionStarted = 0;
    int isColor = 0;
    XI_IMG image;

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE));
    CE(xiSetParamInt(cam, XI_PRM_AEAG, XI_ON));
    CE(xiGetParamInt(cam, XI_PRM_IMAGE_IS_COLOR, &isColor));

    if (isColor)
    {
        CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24));
        CE(xiSetParamInt(cam, XI_PRM_AUTO_WB, XI_ON));
        CE(xiSetParamFloat(cam, XI_PRM_EXP_PRIORITY, 1.0f));
    }
    else
    {
        CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8));
    }

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    CE(xiStartAcquisition(cam));
    isAcquisitionStarted = 1;

    for (int i = 0; i < autoExposureFrameCount; i++)
        CE(xiGetImage(cam, grabTimeoutMs, &image));

    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];
        int filenameLength;

        CE(xiGetImage(cam, grabTimeoutMs, &image));

        filenameLength = snprintf(filename, sizeof(filename), "image%03d.tif", i);
        if (filenameLength < 0 || (size_t)filenameLength >= sizeof(filename) ||
            writeTiffImage(&image, filename) != 0)
        {
            fprintf(stderr, "Error: could not save frame %d as TIFF\n", i + 1);
            return cleanupCamera(cam, isAcquisitionStarted, EXIT_FAILURE);
        }

        printf("Saved frame %d/%d to %s\n", i + 1, frameCount, filename);
    }

    return cleanupCamera(cam, isAcquisitionStarted, EXIT_SUCCESS);
}
