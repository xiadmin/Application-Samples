#pragma once

#include <cstddef>

namespace sample
{

constexpr unsigned int histogramBinCount = 64;

// Counts byteCount 8-bit values into 64 bins of 4 grey levels each and leaves
// the result in p_deviceHistogram (histogramBinCount counters). Both pointers
// must be usable by the GPU: device, managed or mapped host memory. The launch
// is asynchronous; a following cudaMemcpy of the result waits for it.
void computeHistogram64(const unsigned char* p_deviceData, std::size_t byteCount, unsigned int* p_deviceHistogram);

} // namespace sample
