/*
 * Sample name: Capture-10-images-to-file
 * Category: Basic acquisition / image capture
 * OS platform: Cross-Platform
 * Hardware platform: Cross-platform
 * API type: xiAPI
 * Short description: Captures 10 images and saves them as TIFF files using libtiff.
 */

#include <errno.h>
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

static int cleanupCamera(HANDLE p_cam, int isAcquisitionStarted, int result)
{
    XI_RETURN stat;

    if (isAcquisitionStarted)
    {
        stat = xiStopAcquisition(p_cam);
        if (stat != XI_OK)
        {
            fprintf(stderr, "Error: xiStopAcquisition returned %d\n", (int)stat);
            result = EXIT_FAILURE;
        }
    }

    if (p_cam)
    {
        stat = xiCloseDevice(p_cam);
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
            return cleanupCamera(p_cam, isAcquisitionStarted, EXIT_FAILURE); \
        }                                                                     \
    } while (0)

/* Flush, close, and conditionally delete a TIFF file.
 * If writeResult is 0, flushes first; a flush failure is treated as a write
 * error.  TIFFClose is called unconditionally.  When the effective result is
 * non-zero the file is removed; a remove() failure is reported via
 * strerror/errno without overwriting the original error result.
 * Returns the effective result (0 on success, -1 on any failure). */
static int finalizeTiff(TIFF *p_tif, const char *p_filename, int writeResult)
{
    int result = writeResult;

    if (result == 0 && TIFFFlush(p_tif) != 1)
    {
        fprintf(stderr, "Error: TIFFFlush failed for '%s'; output may be incomplete\n",
                p_filename);
        result = -1;
    }

    TIFFClose(p_tif);

    if (result != 0)
    {
        if (remove(p_filename) != 0)
        {
            fprintf(stderr, "Warning: could not remove partial file '%s': %s\n",
                    p_filename, strerror(errno));
        }
    }

    return result;
}

/* Write one XI_IMG to a TIFF file.  Returns 0 on success, -1 on failure.
 * On failure the partially written file (if any) is deleted. */
static int writeTiffImage(const XI_IMG *p_image, const char *p_filename)
{
    uint16_t channelCount;
    uint16_t photometric;
    size_t rowByteCount;
    size_t rowStride;
    TIFF *p_tif = NULL;
    int result = -1;

    if (!p_image || !p_image->bp || !p_filename ||
        p_image->width == 0 || p_image->height == 0)
    {
        fprintf(stderr, "Error: invalid image or filename for TIFF output\n");
        return -1;
    }

    if (p_image->frm == XI_MONO8 || p_image->frm == XI_RAW8)
    {
        channelCount = 1;
        photometric  = PHOTOMETRIC_MINISBLACK;
    }
    else if (p_image->frm == XI_RGB24)
    {
        channelCount = 3;
        photometric  = PHOTOMETRIC_RGB;
    }
    else
    {
        fprintf(stderr, "Error: TIFF mode supports XI_MONO8, XI_RAW8, and XI_RGB24 images\n");
        return -1;
    }

    /* Overflow checks before any arithmetic involving width/height. */
    if ((size_t)p_image->width > SIZE_MAX / channelCount)
    {
        fprintf(stderr, "Error: image width overflows size_t\n");
        return -1;
    }
    rowByteCount = (size_t)p_image->width * channelCount;

    if (rowByteCount > SIZE_MAX - p_image->padding_x)
    {
        fprintf(stderr, "Error: row byte count + padding_x overflows size_t\n");
        return -1;
    }
    rowStride = rowByteCount + p_image->padding_x;

    if ((size_t)p_image->height > SIZE_MAX / rowStride)
    {
        fprintf(stderr, "Error: image dimensions overflow size_t\n");
        return -1;
    }

    /* Validate buffer coverage when the driver reports a size.
     * Minimum bytes needed: last row starts at (height-1)*rowStride and
     * contains rowByteCount valid bytes; padding after the final row is not
     * required.  The overflow is safe because height*rowStride <= SIZE_MAX
     * (checked above) implies (height-1)*rowStride + rowByteCount <= SIZE_MAX. */
    if (p_image->bp_size != 0)
    {
        size_t minBytes = (size_t)(p_image->height - 1) * rowStride + rowByteCount;
        if (minBytes > (size_t)p_image->bp_size)
        {
            fprintf(stderr, "Error: XI_IMG buffer is smaller than the reported image dimensions\n");
            return -1;
        }
    }

    /* libtiff scanline size is tmsize_t (signed); reject widths that overflow it. */
    if (rowByteCount > (size_t)TIFF_TMSIZE_T_MAX)
    {
        fprintf(stderr, "Error: row byte count exceeds libtiff tmsize_t range\n");
        return -1;
    }

    p_tif = TIFFOpen(p_filename, "w");
    if (!p_tif)
    {
        fprintf(stderr, "Error: TIFFOpen failed for '%s'\n", p_filename);
        return -1;
    }

    if (TIFFSetField(p_tif, TIFFTAG_IMAGEWIDTH,      (uint32_t)p_image->width)  != 1 ||
        TIFFSetField(p_tif, TIFFTAG_IMAGELENGTH,     (uint32_t)p_image->height) != 1 ||
        TIFFSetField(p_tif, TIFFTAG_SAMPLESPERPIXEL, channelCount)              != 1 ||
        TIFFSetField(p_tif, TIFFTAG_BITSPERSAMPLE,   (uint16_t)8)               != 1 ||
        TIFFSetField(p_tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT)       != 1 ||
        TIFFSetField(p_tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG)       != 1 ||
        TIFFSetField(p_tif, TIFFTAG_PHOTOMETRIC,     photometric)               != 1 ||
        TIFFSetField(p_tif, TIFFTAG_COMPRESSION,     COMPRESSION_NONE)          != 1 ||
        TIFFSetField(p_tif, TIFFTAG_ROWSPERSTRIP,    (uint32_t)p_image->height) != 1)
    {
        fprintf(stderr, "Error: failed to set TIFF tags for '%s'\n", p_filename);
        return finalizeTiff(p_tif, p_filename, -1);
    }

    if (channelCount == 1)
    {
        /* MONO8/RAW8: pass source rows directly — no copy needed. */
        result = 0;
        for (uint32_t y = 0; y < (uint32_t)p_image->height; y++)
        {
            /* Cast away const: TIFFWriteScanline takes void*, does not modify the buffer. */
            void *p_row = (void *)((const uint8_t *)p_image->bp + (size_t)y * rowStride);
            if (TIFFWriteScanline(p_tif, p_row, y, 0) < 0)
            {
                fprintf(stderr, "Error: TIFFWriteScanline failed at row %u of '%s'\n",
                        y, p_filename);
                result = -1;
                break;
            }
        }
    }
    else
    {
        /* RGB24: allocate one conversion row for BGR->RGB swap. */
        uint8_t *p_rowBuf = (uint8_t *)malloc(rowByteCount);
        if (!p_rowBuf)
        {
            fprintf(stderr, "Error: could not allocate TIFF row buffer\n");
            return finalizeTiff(p_tif, p_filename, -1);
        }

        result = 0;
        for (uint32_t y = 0; y < (uint32_t)p_image->height; y++)
        {
            const uint8_t *p_src = (const uint8_t *)p_image->bp + (size_t)y * rowStride;
            for (uint32_t x = 0; x < (uint32_t)p_image->width; x++)
            {
                p_rowBuf[x * 3u]      = p_src[x * 3u + 2u];
                p_rowBuf[x * 3u + 1u] = p_src[x * 3u + 1u];
                p_rowBuf[x * 3u + 2u] = p_src[x * 3u];
            }
            if (TIFFWriteScanline(p_tif, p_rowBuf, y, 0) < 0)
            {
                fprintf(stderr, "Error: TIFFWriteScanline failed at row %u of '%s'\n",
                        y, p_filename);
                result = -1;
                break;
            }
        }

        free(p_rowBuf);
    }

    return finalizeTiff(p_tif, p_filename, result);
}

int main(void)
{
    HANDLE p_cam = NULL;
    int isAcquisitionStarted = 0;
    int isColor = 0;
    XI_IMG image;

    printf("Opening camera index 0\n");
    CE(xiOpenDevice(0, &p_cam));
    CE(xiSetParamInt(p_cam, XI_PRM_EXPOSURE, exposureUs));
    CE(xiSetParamInt(p_cam, XI_PRM_BUFFER_POLICY, XI_BP_SAFE));
    CE(xiSetParamInt(p_cam, XI_PRM_AEAG, XI_ON));
    CE(xiGetParamInt(p_cam, XI_PRM_IMAGE_IS_COLOR, &isColor));

    if (isColor)
    {
        CE(xiSetParamInt(p_cam, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24));
        CE(xiSetParamInt(p_cam, XI_PRM_AUTO_WB, XI_ON));
        CE(xiSetParamFloat(p_cam, XI_PRM_EXP_PRIORITY, 1.0f));
    }
    else
    {
        CE(xiSetParamInt(p_cam, XI_PRM_IMAGE_DATA_FORMAT, XI_MONO8));
    }

    memset(&image, 0, sizeof(image));
    image.size = sizeof(image);

    CE(xiStartAcquisition(p_cam));
    isAcquisitionStarted = 1;

    for (int i = 0; i < autoExposureFrameCount; i++)
        CE(xiGetImage(p_cam, grabTimeoutMs, &image));

    for (int i = 0; i < frameCount; i++)
    {
        char filename[32];
        int filenameLength;

        CE(xiGetImage(p_cam, grabTimeoutMs, &image));

        filenameLength = snprintf(filename, sizeof(filename), "image%03d.tif", i);
        if (filenameLength < 0 || (size_t)filenameLength >= sizeof(filename) ||
            writeTiffImage(&image, filename) != 0)
        {
            fprintf(stderr, "Error: could not save frame %d as TIFF\n", i + 1);
            return cleanupCamera(p_cam, isAcquisitionStarted, EXIT_FAILURE);
        }

        printf("Saved frame %d/%d to %s\n", i + 1, frameCount, filename);
    }

    return cleanupCamera(p_cam, isAcquisitionStarted, EXIT_SUCCESS);
}
