/*
 * Sample name: Capture-10-images-to-ram
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images into application-owned RAM.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <xiApi.h>

static const int frameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;

static int cleanupSample(HANDLE cam, int isAcquisitionStarted, XI_IMG *images,
                         unsigned char *storage, int result)
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

    free(storage);
    free(images);

    if (result == EXIT_SUCCESS)
        printf("Done\n");

    return result;
}

#define CE(func)                                                               \
    do                                                                         \
    {                                                                          \
        XI_RETURN stat = (func);                                               \
        if (stat != XI_OK)                                                     \
        {                                                                      \
            fprintf(stderr, "Error: %s returned %d\n", #func, (int)stat);    \
            return cleanupSample(cam, isAcquisitionStarted, images, storage,   \
                                 EXIT_FAILURE);                                \
        }                                                                      \
    } while (0)

int main(void)
{
    HANDLE cam = NULL;
    int isAcquisitionStarted = 0;
    int payloadSize = 0;
    XI_IMG *images = NULL;
    unsigned char *storage = NULL;

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8));
    CE(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiGetParamInt(cam, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize));

    if (payloadSize <= 0 || (size_t)payloadSize > SIZE_MAX / (size_t)frameCount)
    {
        fprintf(stderr, "Error: invalid image payload size %d\n", payloadSize);
        return cleanupSample(cam, isAcquisitionStarted, images, storage, EXIT_FAILURE);
    }

    images = calloc((size_t)frameCount, sizeof(*images));
    storage = malloc((size_t)payloadSize * (size_t)frameCount);
    if (!images || !storage)
    {
        fprintf(stderr, "Error: could not allocate RAM for %d images\n", frameCount);
        return cleanupSample(cam, isAcquisitionStarted, images, storage, EXIT_FAILURE);
    }

    for (int i = 0; i < frameCount; i++)
    {
        images[i].size = sizeof(images[i]);
        images[i].bp = storage + (size_t)i * (size_t)payloadSize;
        images[i].bp_size = (uint32_t)payloadSize;
    }

    CE(xiStartAcquisition(cam));
    isAcquisitionStarted = 1;

    for (int i = 0; i < frameCount; i++)
        CE(xiGetImage(cam, grabTimeoutMs, &images[i]));

    CE(xiStopAcquisition(cam));
    isAcquisitionStarted = 0;
    CE(xiCloseDevice(cam));
    cam = NULL;

    printf("Camera closed; reading retained images from RAM\n");
    for (int i = 0; i < frameCount; i++)
    {
        printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n", i + 1, frameCount,
               (unsigned)images[i].width, (unsigned)images[i].height,
               (unsigned)images[i].nframe,
               images[i].bp ? (int)((const unsigned char *)images[i].bp)[0] : -1);
    }

    return cleanupSample(cam, isAcquisitionStarted, images, storage, EXIT_SUCCESS);
}
