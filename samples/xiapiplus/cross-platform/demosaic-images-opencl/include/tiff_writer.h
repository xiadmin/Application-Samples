#pragma once

#include <cstdint>
#include <string>

namespace sample
{

// Writes interleaved 8-bit RGB pixels (width * height * 3 bytes, no row
// padding) as an uncompressed TIFF file. Throws std::runtime_error on failure.
void writeRgb8Tiff(const std::string& path, const std::uint8_t* p_pixels, std::uint32_t width, std::uint32_t height);

} // namespace sample
