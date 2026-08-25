#ifndef TIFF_WRITER_H
#define TIFF_WRITER_H

#include <xiApi.h>

/**
 * Writes an 8-bit mono/RAW or RGB24 XI_IMG as an uncompressed baseline TIFF.
 * Returns zero on success and a nonzero value on failure.
 */
int writeTiffImage(const XI_IMG *image, const char *filename);

#endif
