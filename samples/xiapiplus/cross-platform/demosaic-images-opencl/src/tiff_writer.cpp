#include <cstddef>
#include <memory>
#include <stdexcept>

#include <tiffio.h>

#include "tiff_writer.h"

namespace sample
{

void writeRgb8Tiff(const std::string& path, const std::uint8_t* p_pixels, std::uint32_t width, std::uint32_t height)
{
    std::unique_ptr<TIFF, decltype(&TIFFClose)> p_tiff(TIFFOpen(path.c_str(), "w"), &TIFFClose);
    if (!p_tiff)
        throw std::runtime_error("TIFFOpen failed for " + path);

    const bool isHeaderWritten =
        TIFFSetField(p_tiff.get(), TIFFTAG_IMAGEWIDTH, width) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_IMAGELENGTH, height) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_SAMPLESPERPIXEL, static_cast<std::uint16_t>(3)) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_BITSPERSAMPLE, static_cast<std::uint16_t>(8)) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_COMPRESSION, COMPRESSION_NONE) == 1 &&
        TIFFSetField(p_tiff.get(), TIFFTAG_ROWSPERSTRIP, height) == 1;
    if (!isHeaderWritten)
        throw std::runtime_error("TIFFSetField failed for " + path);

    const std::size_t rowBytes = static_cast<std::size_t>(width) * 3;
    for (std::uint32_t row = 0; row < height; ++row)
    {
        // TIFFWriteScanline() takes a non-const pointer but does not modify the row.
        auto* p_row = const_cast<std::uint8_t*>(p_pixels + row * rowBytes);
        if (TIFFWriteScanline(p_tiff.get(), p_row, row, 0) < 0)
            throw std::runtime_error("TIFFWriteScanline failed for " + path);
    }
}

} // namespace sample
