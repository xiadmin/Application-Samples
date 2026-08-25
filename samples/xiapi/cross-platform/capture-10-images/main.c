/*
 * Sample name: Capture-10-images
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images for printing, TIFF output, or retention in RAM.
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xiApi.h>

#include "tiff_writer.h"

static const int frameCount = 10;
static const int autoExposureFrameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;

typedef enum
{
    MODE_PRINT,
    MODE_TIFF,
    MODE_RAM
} CaptureMode;

static void printUsage(const char *programName)
{
    printf("Usage: %s [--mode print|tiff|ram]\n", programName);
    printf("Options:\n");
    printf("  --help                 Show this help.\n");
    printf("  --mode print|tiff|ram  Select the capture behavior.\n");
}

static int checkXi(XI_RETURN stat, const char *operation)
{
    if (stat == XI_OK)
        return EXIT_SUCCESS;

    fprintf(stderr, "Error: %s returned %d\n", operation, (int)stat);
    return EXIT_FAILURE;
}

static int cleanupCamera(HANDLE cam, int result)
{
    if (cam)
    {
        XI_RETURN stopStatus = xiStopAcquisition(cam);
        if (result == EXIT_SUCCESS && checkXi(stopStatus, "xiStopAcquisition") != EXIT_SUCCESS)
            result = EXIT_FAILURE;

        if (checkXi(xiCloseDevice(cam), "xiCloseDevice") != EXIT_SUCCESS)
            result = EXIT_FAILURE;
    }

    if (result == EXIT_SUCCESS)
        printf("Done\n");

    return result;
}

static void printFrame(const XI_IMG *image, int index)
{
    printf("Frame %d/%d: %ux%u nframe=%u first_byte=%d\n", index + 1, frameCount,
           (unsigned)image->width, (unsigned)image->height, (unsigned)image->nframe,
           image->bp ? (int)((const unsigned char *)image->bp)[0] : -1);
}

static int runPrintMode(HANDLE cam)
{
    XI_IMG image;

    if (checkXi(xiStartAcquisition(cam), "xiStartAcquisition") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    printf("Capturing %d frames in print mode\n", frameCount);

    for (int i = 0; i < frameCount; i++)
    {
        memset(&image, 0, sizeof(image));
        image.size = sizeof(image);

        if (checkXi(xiGetImage(cam, grabTimeoutMs, &image), "xiGetImage") != EXIT_SUCCESS)
            return EXIT_FAILURE;

        printFrame(&image, i);
    }

    return EXIT_SUCCESS;
}

static int configureTiffMode(HANDLE cam)
{
    int isColor = 0;

    if (checkXi(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE),
                "xiSetParamInt(XI_PRM_BUFFER_POLICY)") != EXIT_SUCCESS ||
        checkXi(xiSetParamInt(cam, XI_PRM_AEAG, XI_ON),
                "xiSetParamInt(XI_PRM_AEAG)") != EXIT_SUCCESS ||
        checkXi(xiGetParamInt(cam, XI_PRM_IMAGE_IS_COLOR, &isColor),
                "xiGetParamInt(XI_PRM_IMAGE_IS_COLOR)") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (isColor)
    {
        if (checkXi(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24),
                    "xiSetParamInt(XI_PRM_IMAGE_DATA_FORMAT)") != EXIT_SUCCESS ||
            checkXi(xiSetParamInt(cam, XI_PRM_AUTO_WB, XI_ON),
                    "xiSetParamInt(XI_PRM_AUTO_WB)") != EXIT_SUCCESS ||
            checkXi(xiSetParamFloat(cam, XI_PRM_EXP_PRIORITY, 1.0f),
                    "xiSetParamFloat(XI_PRM_EXP_PRIORITY)") != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    else if (checkXi(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8),
                     "xiSetParamInt(XI_PRM_IMAGE_DATA_FORMAT)") != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static int runTiffMode(HANDLE cam)
{
    XI_IMG image;

    if (configureTiffMode(cam) != EXIT_SUCCESS)
        return EXIT_FAILURE;

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    /* Reuse one safe destination because each frame is saved before the next xiGetImage call. */

    if (checkXi(xiStartAcquisition(cam), "xiStartAcquisition") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    printf("Warming up auto exposure for %d frames\n", autoExposureFrameCount);
    for (int i = 0; i < autoExposureFrameCount; i++)
    {
        if (checkXi(xiGetImage(cam, grabTimeoutMs, &image), "xiGetImage") != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }

    printf("Capturing %d frames in TIFF mode\n", frameCount);
    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];

        if (checkXi(xiGetImage(cam, grabTimeoutMs, &image), "xiGetImage") != EXIT_SUCCESS)
            return EXIT_FAILURE;

        if (snprintf(filename, sizeof(filename), "image%03d.tif", i) < 0 ||
            writeTiffImage(&image, filename) != 0)
        {
            fprintf(stderr, "Error: could not save frame %d as TIFF\n", i + 1);
            return EXIT_FAILURE;
        }

        printf("Saved frame %d/%d to %s\n", i + 1, frameCount, filename);
    }

    return EXIT_SUCCESS;
}

static int runRamMode(HANDLE *camPtr)
{
    HANDLE cam = *camPtr;
    XI_IMG *images = NULL;
    unsigned char *storage = NULL;
    int payloadSize = 0;
    int result = EXIT_FAILURE;

    if (checkXi(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8),
                "xiSetParamInt(XI_PRM_IMAGE_DATA_FORMAT)") != EXIT_SUCCESS ||
        checkXi(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE),
                "xiSetParamInt(XI_PRM_BUFFER_POLICY)") != EXIT_SUCCESS ||
        checkXi(xiGetParamInt(cam, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize),
                "xiGetParamInt(XI_PRM_IMAGE_PAYLOAD_SIZE)") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (payloadSize <= 0 || (size_t)payloadSize > SIZE_MAX / (size_t)frameCount)
    {
        fprintf(stderr, "Error: invalid image payload size %d\n", payloadSize);
        return EXIT_FAILURE;
    }

    images = calloc((size_t)frameCount, sizeof(*images));
    storage = malloc((size_t)payloadSize * (size_t)frameCount);
    if (!images || !storage)
    {
        fprintf(stderr, "Error: could not allocate RAM for %d images\n", frameCount);
        free(storage);
        free(images);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < frameCount; i++)
    {
        images[i].size = sizeof(images[i]);
        images[i].bp = storage + (size_t)i * (size_t)payloadSize;
        images[i].bp_size = (uint32_t)payloadSize;
    }

    if (checkXi(xiStartAcquisition(cam), "xiStartAcquisition") == EXIT_SUCCESS)
    {
        result = EXIT_SUCCESS;
        printf("Capturing %d frames into application-owned RAM\n", frameCount);

        for (int i = 0; i < frameCount; i++)
        {
            if (checkXi(xiGetImage(cam, grabTimeoutMs, &images[i]), "xiGetImage") != EXIT_SUCCESS)
            {
                result = EXIT_FAILURE;
                break;
            }
            printFrame(&images[i], i);
        }
    }

    if (result == EXIT_SUCCESS &&
        checkXi(xiStopAcquisition(cam), "xiStopAcquisition") == EXIT_SUCCESS &&
        checkXi(xiCloseDevice(cam), "xiCloseDevice") == EXIT_SUCCESS)
    {
        *camPtr = NULL;
        printf("Camera closed; reading retained images from RAM\n");
        for (int i = 0; i < frameCount; i++)
            printFrame(&images[i], i);
    }
    else
    {
        result = EXIT_FAILURE;
    }

    free(storage);
    free(images);
    return result;
}

int main(int argc, char *argv[])
{
    uint32_t count = 0;
    HANDLE cam = NULL;
    CaptureMode mode = MODE_PRINT;
    int result;

    if (argc == 2 && strcmp(argv[1], "--help") == 0)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (argc == 3 && strcmp(argv[1], "--mode") == 0)
    {
        if (strcmp(argv[2], "print") == 0)
            mode = MODE_PRINT;
        else if (strcmp(argv[2], "tiff") == 0)
            mode = MODE_TIFF;
        else if (strcmp(argv[2], "ram") == 0)
            mode = MODE_RAM;
        else
        {
            fprintf(stderr, "Error: invalid mode '%s'\n", argv[2]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    else if (argc != 1)
    {
        fprintf(stderr, "Error: invalid arguments\n");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (checkXi(xiGetNumberDevices(&count), "xiGetNumberDevices") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (count == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Found %u camera(s), opening index 0\n", (unsigned)count);

    if (checkXi(xiOpenDevice(0, &cam), "xiOpenDevice") != EXIT_SUCCESS)
        return EXIT_FAILURE;

    if (checkXi(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs),
                "xiSetParamInt(XI_PRM_EXPOSURE)") != EXIT_SUCCESS)
        return cleanupCamera(cam, EXIT_FAILURE);

    printf("Exposure: %d us (%d ms)\n", exposureUs, exposureUs / 1000);

    if (mode == MODE_TIFF)
        result = runTiffMode(cam);
    else if (mode == MODE_RAM)
        result = runRamMode(&cam);
    else
        result = runPrintMode(cam);

    return cleanupCamera(cam, result);
}
