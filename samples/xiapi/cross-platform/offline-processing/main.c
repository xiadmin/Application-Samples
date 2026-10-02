/*
 * Sample name: Offline-processing
 * Category: Image processing
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 RAW8 images, converts them to RGB24 with
 * xiAPI offline processing after acquisition, and saves them as TIFF files.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tiffio.h>

#include <xiApi.h>

static const int frameCount = 10;
static const int exposureUs = 10000;
static const int grabTimeoutMs = 5000;
static const int processTimeoutMs = 5000;

/* Check error macro. Executes the call, reports and leaves main if it fails.
 * The camera handle, the processing handle and the RAM buffers are released by
 * the driver and the OS when the process exits, so the error path does not
 * unwind explicitly. */
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

/* Write one XI_IMG to an uncompressed TIFF file. Handles the two formats the
 * offline processing produces here: XI_MONO8 and XI_RGB24. Returns 0 on success, -1 on failure. */
static int writeTiffImage(const XI_IMG *p_image, const char *p_filename)
{
    int isColor = (p_image->frm == XI_RGB24);
    int channelCount = isColor ? 3 : 1;
    /* The driver may pad each line, so the source stride is wider than the row. */
    size_t rowByteCount = (size_t)p_image->width * channelCount;
    size_t rowStride = rowByteCount + p_image->padding_x;
    unsigned char *p_row = NULL;
    TIFF *p_tif = NULL;
    int result = 0;

    p_row = (unsigned char *)malloc(rowByteCount);
    if (!p_row)
    {
        fprintf(stderr, "Error: could not allocate TIFF row buffer\n");
        return -1;
    }

    p_tif = TIFFOpen(p_filename, "w");
    if (!p_tif)
    {
        fprintf(stderr, "Error: TIFFOpen failed for '%s'\n", p_filename);
        free(p_row);
        return -1;
    }

    if (TIFFSetField(p_tif, TIFFTAG_IMAGEWIDTH,      (uint32_t)p_image->width)  != 1 ||
        TIFFSetField(p_tif, TIFFTAG_IMAGELENGTH,     (uint32_t)p_image->height) != 1 ||
        TIFFSetField(p_tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)channelCount)    != 1 ||
        TIFFSetField(p_tif, TIFFTAG_BITSPERSAMPLE,   (uint16_t)8)               != 1 ||
        TIFFSetField(p_tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT)       != 1 ||
        TIFFSetField(p_tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG)       != 1 ||
        TIFFSetField(p_tif, TIFFTAG_PHOTOMETRIC,
                     isColor ? PHOTOMETRIC_RGB : PHOTOMETRIC_MINISBLACK)        != 1 ||
        TIFFSetField(p_tif, TIFFTAG_COMPRESSION,     COMPRESSION_NONE)          != 1 ||
        TIFFSetField(p_tif, TIFFTAG_ROWSPERSTRIP,    (uint32_t)p_image->height) != 1)
    {
        fprintf(stderr, "Error: failed to set TIFF tags for '%s'\n", p_filename);
        result = -1;
    }

    /* Gating on `result` keeps one cleanup path without an early return. */
    for (uint32_t y = 0; result == 0 && y < (uint32_t)p_image->height; y++)
    {
        const unsigned char *p_src = (const unsigned char *)p_image->bp + (size_t)y * rowStride;

        if (isColor)
        {
            /* xiAPI delivers XI_RGB24 as BGR; TIFF expects RGB. */
            for (uint32_t x = 0; x < (uint32_t)p_image->width; x++)
            {
                p_row[x * 3u]      = p_src[x * 3u + 2u];
                p_row[x * 3u + 1u] = p_src[x * 3u + 1u];
                p_row[x * 3u + 2u] = p_src[x * 3u];
            }
        }
        else
        {
            memcpy(p_row, p_src, rowByteCount);
        }

        if (TIFFWriteScanline(p_tif, p_row, y, 0) < 0)
        {
            fprintf(stderr, "Error: TIFFWriteScanline failed at row %u of '%s'\n", y, p_filename);
            result = -1;
            break;
        }
    }

    TIFFClose(p_tif);
    free(p_row);
    return result;
}

int main(void)
{
    HANDLE cam = NULL;
    xiProcessingHandle_t proc = NULL;
    DWORD deviceCount = 0;
    int isColor = 0;
    int payloadSize = 0;
    int contextSize = 0;
    int outputFormat = XI_MONO8;
    unsigned char *p_storage = NULL;
    char *p_context = NULL;
    XI_IMG image;

    CE(xiGetNumberDevices(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "Error: no XIMEA cameras detected\n");
        return EXIT_FAILURE;
    }

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &cam));
    CE(xiSetParamInt(cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8));
    CE(xiSetParamInt(cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE));
    CE(xiSetParamInt(cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiGetParamInt(cam, XI_PRM_IMAGE_IS_COLOR, &isColor));

    if (isColor)
    {
        /* Not applied to the RAW8 frames; recorded in the context list below
         * and applied by the offline processing. */
        CE(xiSetParamFloat(cam, XI_PRM_WB_KR, 1.05f));
        CE(xiSetParamFloat(cam, XI_PRM_WB_KG, 1.0f));
        CE(xiSetParamFloat(cam, XI_PRM_WB_KB, 2.10f));
        outputFormat = XI_RGB24;
    }

    CE(xiGetParamInt(cam, XI_PRM_IMAGE_PAYLOAD_SIZE, &payloadSize));
    if (payloadSize <= 0 || (size_t)payloadSize > SIZE_MAX / (size_t)frameCount)
    {
        fprintf(stderr, "Error: invalid image payload size %d\n", payloadSize);
        return EXIT_FAILURE;
    }

    p_storage = malloc((size_t)payloadSize * (size_t)frameCount);
    if (!p_storage)
    {
        fprintf(stderr, "Error: could not allocate RAM for %d images\n", frameCount);
        return EXIT_FAILURE;
    }

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    CE(xiStartAcquisition(cam));

    /* XI_BP_SAFE copies each frame into the buffer pointed to by bp. */
    for (int i = 0; i < frameCount; i++)
    {
        image.bp = p_storage + (size_t)i * (size_t)payloadSize;
        image.bp_size = (DWORD)payloadSize;
        CE(xiGetImage(cam, grabTimeoutMs, &image));
    }

    CE(xiStopAcquisition(cam));

    /* The context list holds every setting the frames were captured with. It
     * is all the processing needs, so the camera can be closed afterwards. */
    CE(xiGetParamInt(cam, XI_PRM_API_CONTEXT_LIST XI_PRMM_REQ_VAL_BUFFER_SIZE, &contextSize));
    if (contextSize <= 0)
    {
        fprintf(stderr, "Error: invalid context list size %d\n", contextSize);
        return EXIT_FAILURE;
    }

    p_context = malloc((size_t)contextSize);
    if (!p_context)
    {
        fprintf(stderr, "Error: could not allocate %d bytes for the context list\n", contextSize);
        return EXIT_FAILURE;
    }

    CE(xiGetParamString(cam, XI_PRM_API_CONTEXT_LIST, p_context, (DWORD)contextSize));
    CE(xiCloseDevice(cam));

    printf("Camera closed; processing %d stored frames offline\n", frameCount);
    CE(xiProcOpen(&proc));
    CE(xiProcSetParam(proc, XI_PRM_API_CONTEXT_LIST, p_context, (DWORD)strlen(p_context), xiTypeString));
    CE(xiProcSetParam(proc, XI_PRM_IMAGE_DATA_FORMAT, &outputFormat, sizeof(outputFormat), xiTypeInteger));

    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];
        XI_IMG processed;

        /* bp == NULL: the processing owns the output buffer until the next pull. */
        memset(&processed, 0, sizeof(processed));
        processed.size = sizeof(processed);

        CE(xiProcPushImage(proc, p_storage + (size_t)i * (size_t)payloadSize));
        CE(xiProcPullImage(proc, processTimeoutMs, &processed));

        snprintf(filename, sizeof(filename), "image%03d.tif", i);
        if (writeTiffImage(&processed, filename) != 0)
        {
            fprintf(stderr, "Error: could not save frame %d as TIFF\n", i + 1);
            return EXIT_FAILURE;
        }

        printf("Processed frame %d/%d (%ux%u) to %s\n", i + 1, frameCount,
               (unsigned)processed.width, (unsigned)processed.height, filename);
    }

    CE(xiProcClose(proc));
    free(p_context);
    free(p_storage);

    printf("Done\n");
    return EXIT_SUCCESS;
}
