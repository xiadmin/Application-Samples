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

/* Check error macro. Executes the call, reports and leaves main if it fails.
 * The camera handle and the RAM buffers are released by the driver and the OS
 * when the process exits, so the error path does not unwind explicitly. */
#define CE(func)                                                               \
    do                                                                         \
    {                                                                          \
        XI_RETURN stat = (func);                                               \
        if (stat != XI_OK)                                                     \
        {                                                                      \
            fprintf(stderr, "Error: %s returned %d\n", #func, (int)stat);      \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void)
{
    HANDLE cam = NULL;
    int payloadSize = 0;
    XI_IMG *p_images = NULL;
    unsigned char *p_storage = NULL;

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8));
    CE(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiGetParamInt(cam, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize));

    /* All frames are held at once, so the total size must not overflow size_t. */
    if (payloadSize <= 0 || (size_t)payloadSize > SIZE_MAX / (size_t)frameCount)
    {
        fprintf(stderr, "Error: invalid image payload size %d\n", payloadSize);
        return EXIT_FAILURE;
    }

    p_images = calloc((size_t)frameCount, sizeof(*p_images));
    p_storage = malloc((size_t)payloadSize * (size_t)frameCount);
    if (!p_images || !p_storage)
    {
        fprintf(stderr, "Error: could not allocate RAM for %d images\n", frameCount);
        return EXIT_FAILURE;
    }

    /* XI_BP_SAFE copies each frame into the buffer pointed to by bp. */
    for (int i = 0; i < frameCount; i++)
    {
        p_images[i].size = sizeof(p_images[i]);
        p_images[i].bp = p_storage + (size_t)i * (size_t)payloadSize;
        p_images[i].bp_size = (uint32_t)payloadSize;
    }

    CE(xiStartAcquisition(cam));

    for (int i = 0; i < frameCount; i++)
        CE(xiGetImage(cam, grabTimeoutMs, &p_images[i]));

    CE(xiStopAcquisition(cam));
    CE(xiCloseDevice(cam));

    printf("Camera closed; reading retained images from RAM\n");
    for (int i = 0; i < frameCount; i++)
    {
        printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n", i + 1, frameCount,
               (unsigned)p_images[i].width, (unsigned)p_images[i].height,
               (unsigned)p_images[i].nframe,
               p_images[i].bp ? (int)((const unsigned char *)p_images[i].bp)[0] : -1);
    }

    free(p_storage);
    free(p_images);

    printf("Done\n");
    return EXIT_SUCCESS;
}
