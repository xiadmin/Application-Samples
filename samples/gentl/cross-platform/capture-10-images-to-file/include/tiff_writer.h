#pragma once

#include <cstdint>
#include <string>

namespace sample
{

// Writes an 8-bit single-channel image as an uncompressed TIFF file.
// Throws std::runtime_error when libtiff reports a failure.
void writeMono8Tiff(const std::string& path, const std::uint8_t* p_pixels, std::uint32_t width,
                    std::uint32_t height);

} // namespace sample
