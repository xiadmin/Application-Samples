#include "tiff_writer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TIFF_TYPE_SHORT 3u
#define TIFF_TYPE_LONG 4u

static int writeU16(FILE *file, uint16_t value)
{
    unsigned char bytes[2] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int writeU32(FILE *file, uint32_t value)
{
    unsigned char bytes[4] = {
        (unsigned char)(value & 0xffu),
        (unsigned char)((value >> 8) & 0xffu),
        (unsigned char)((value >> 16) & 0xffu),
        (unsigned char)((value >> 24) & 0xffu)};
    return fwrite(bytes, 1, sizeof(bytes), file) == sizeof(bytes) ? 0 : -1;
}

static int writeEntry(FILE *file, uint16_t tag, uint16_t type, uint32_t count, uint32_t value)
{
    if (writeU16(file, tag) != 0 || writeU16(file, type) != 0 || writeU32(file, count) != 0)
        return -1;

    if (type == TIFF_TYPE_SHORT && count == 1)
        return writeU16(file, (uint16_t)value) != 0 || writeU16(file, 0) != 0 ? -1 : 0;

    return writeU32(file, value);
}

static int writeMetadata(FILE *file, const XI_IMG *image, uint16_t entryCount,
                         uint32_t ifdOffset, uint32_t bitsOffset, uint32_t pixelOffset,
                         uint32_t pixelByteCount, unsigned int channelCount,
                         unsigned int photometric, const char *filename)
{
    if (fwrite("II", 1, 2, file) != 2 || writeU16(file, 42) != 0 ||
        writeU32(file, ifdOffset) != 0 || writeU16(file, entryCount) != 0 ||
        writeEntry(file, 256, TIFF_TYPE_LONG, 1, image->width) != 0 ||
        writeEntry(file, 257, TIFF_TYPE_LONG, 1, image->height) != 0 ||
        writeEntry(file, 258, TIFF_TYPE_SHORT, channelCount,
                   channelCount == 1 ? 8u : bitsOffset) != 0 ||
        writeEntry(file, 259, TIFF_TYPE_SHORT, 1, 1) != 0 ||
        writeEntry(file, 262, TIFF_TYPE_SHORT, 1, photometric) != 0 ||
        writeEntry(file, 273, TIFF_TYPE_LONG, 1, pixelOffset) != 0 ||
        writeEntry(file, 277, TIFF_TYPE_SHORT, 1, channelCount) != 0 ||
        writeEntry(file, 278, TIFF_TYPE_LONG, 1, image->height) != 0 ||
        writeEntry(file, 279, TIFF_TYPE_LONG, 1, pixelByteCount) != 0 ||
        writeU32(file, 0) != 0)
    {
        fprintf(stderr, "Error: could not write TIFF metadata to '%s'\n", filename);
        return -1;
    }

    if (channelCount == 3 &&
        (writeU16(file, 8) != 0 || writeU16(file, 8) != 0 || writeU16(file, 8) != 0))
    {
        fprintf(stderr, "Error: could not write TIFF channel metadata to '%s'\n", filename);
        return -1;
    }

    return 0;
}

static int writePixelData(FILE *file, const XI_IMG *image, size_t rowByteCount,
                          size_t rowStride, unsigned int channelCount, const char *filename)
{
    unsigned char *rgbRow = NULL;

    if (channelCount == 3)
    {
        rgbRow = malloc(rowByteCount);
        if (!rgbRow)
        {
            fprintf(stderr, "Error: could not allocate a TIFF row buffer\n");
            return -1;
        }
    }

    for (uint32_t y = 0; y < image->height; y++)
    {
        const unsigned char *source = (const unsigned char *)image->bp + (size_t)y * rowStride;
        const unsigned char *output = source;

        if (channelCount == 3)
        {
            for (uint32_t x = 0; x < image->width; x++)
            {
                rgbRow[x * 3u] = source[x * 3u + 2u];
                rgbRow[x * 3u + 1u] = source[x * 3u + 1u];
                rgbRow[x * 3u + 2u] = source[x * 3u];
            }
            output = rgbRow;
        }

        if (fwrite(output, 1, rowByteCount, file) != rowByteCount)
        {
            fprintf(stderr, "Error: could not write TIFF pixel data to '%s'\n", filename);
            free(rgbRow);
            return -1;
        }
    }

    free(rgbRow);
    return 0;
}

int writeTiffImage(const XI_IMG *image, const char *filename)
{
    const uint16_t entryCount = 9;
    const uint32_t ifdOffset = 8;
    const uint32_t ifdSize = 2u + (uint32_t)entryCount * 12u + 4u;
    uint32_t bitsOffset;
    uint32_t pixelOffset;
    uint32_t pixelByteCount;
    size_t rowByteCount;
    size_t rowStride;
    unsigned int channelCount;
    unsigned int photometric;
    FILE *file = NULL;
    int result = -1;

    if (!image || !image->bp || !filename || image->width == 0 || image->height == 0)
    {
        fprintf(stderr, "Error: invalid image or filename for TIFF output\n");
        return -1;
    }

    if (image->frm == XI_MONO8 || image->frm == XI_RAW8)
    {
        channelCount = 1;
        photometric = 1;
    }
    else if (image->frm == XI_RGB24)
    {
        channelCount = 3;
        photometric = 2;
    }
    else
    {
        fprintf(stderr, "Error: TIFF mode supports XI_MONO8, XI_RAW8, and XI_RGB24 images\n");
        return -1;
    }

    if ((size_t)image->width > SIZE_MAX / channelCount)
        return -1;
    rowByteCount = (size_t)image->width * channelCount;
    if (rowByteCount > SIZE_MAX - image->padding_x)
        return -1;
    rowStride = rowByteCount + image->padding_x;
    if ((size_t)image->height > SIZE_MAX / rowStride ||
        (size_t)image->height > SIZE_MAX / rowByteCount)
        return -1;

    /* Some driver-owned xiAPI buffers report bp_size as zero (unknown). */
    if (image->bp_size != 0 && rowStride * image->height > image->bp_size)
    {
        fprintf(stderr, "Error: XI_IMG buffer is smaller than the reported image dimensions\n");
        return -1;
    }

    bitsOffset = ifdOffset + ifdSize;
    pixelOffset = bitsOffset + (channelCount == 3 ? 6u : 0u);
    if ((size_t)image->height > (UINT32_MAX - pixelOffset) / rowByteCount)
    {
        fprintf(stderr, "Error: image is too large for baseline TIFF output\n");
        return -1;
    }
    pixelByteCount = (uint32_t)(rowByteCount * image->height);

    file = fopen(filename, "wb");
    if (!file)
    {
        fprintf(stderr, "Error: could not open TIFF output file '%s'\n", filename);
        return -1;
    }

    if (writeMetadata(file, image, entryCount, ifdOffset, bitsOffset, pixelOffset,
                      pixelByteCount, channelCount, photometric, filename) == 0 &&
        writePixelData(file, image, rowByteCount, rowStride, channelCount, filename) == 0)
        result = 0;

    if (fclose(file) != 0)
        result = -1;
    if (result != 0)
        remove(filename);
    return result;
}
