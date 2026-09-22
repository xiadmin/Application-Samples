/*
 * Sample name: Capture-10-images-to-file
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images and saves them as TIFF files using libtiff.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tiffio.h>

#include <xiApi.h>

static const int frameCount             = 10;
static const int autoExposureFrameCount = 10;
static const int exposureUs             = 10000;
static const int grabTimeoutMs          = 5000;

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

/* Write one XI_IMG to an uncompressed TIFF file. Handles the two formats this
 * sample requests: XI_MONO8 and XI_RGB24. Returns 0 on success, -1 on failure. */
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

    /* Discard the first frames while auto exposure and white balance settle. */
    for (int i = 0; i < autoExposureFrameCount; i++)
        CE(xiGetImage(cam, grabTimeoutMs, &image));

    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];

        CE(xiGetImage(cam, grabTimeoutMs, &image));

        snprintf(filename, sizeof(filename), "image%03d.tif", i);
        if (writeTiffImage(&image, filename) != 0)
        {
            fprintf(stderr, "Error: could not save frame %d as TIFF\n", i + 1);
            return EXIT_FAILURE;
        }

        printf("Saved frame %d/%d to %s\n", i + 1, frameCount, filename);
    }

    CE(xiStopAcquisition(cam));
    CE(xiCloseDevice(cam));

    printf("Done\n");
    return EXIT_SUCCESS;
}
