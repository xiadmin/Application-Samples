#include "demosaic_kernel.h"

namespace sample
{

const char* const demosaicKernelSource = R"CLC(
// Malvar-He-Cutler demosaicing as formulated by Morgan McGuire, "Efficient,
// High-Quality Bayer Demosaic Filtering on GPUs" (JGT 2008), and ported to
// OpenCL by Ron Woods. Each work item turns one RAW8 Bayer pixel into one
// 8-bit RGB pixel.

// Reflects a coordinate outside the frame back into it by an even distance,
// so the mirrored pixel has the same Bayer color: -1 -> 1, -2 -> 2,
// size -> size - 2.
int mirror(int value, int size)
{
    if (value < 0)
        return -value;
    if (value >= size)
        return 2 * (size - 1) - value;
    return value;
}

__kernel void demosaic(__global const uchar* raw, const int rawStride, __global uchar* rgb,
                       const int redX, const int redY, const float gainRed, const float gainGreen,
                       const float gainBlue)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int width = get_global_size(0);
    const int height = get_global_size(1);

#define FETCH(dx, dy) ((float)raw[mirror(y + (dy), height) * rawStride + mirror(x + (dx), width)])

    // Position in the 2x2 Bayer tile, shifted so that (0, 0) is the red pixel.
    const int columnParity = (x + redX) & 1;
    const int rowParity = (y + redY) & 1;

    const float C = FETCH(0, 0);
    const float D = FETCH(-1, -1) + FETCH(-1, 1) + FETCH(1, -1) + FETCH(1, 1);
    const float A = FETCH(0, -2) + FETCH(0, 2);
    const float B = FETCH(0, -1) + FETCH(0, 1);
    const float E = FETCH(-2, 0) + FETCH(2, 0);
    const float F = FETCH(-1, 0) + FETCH(1, 0);

    const float4 kA = (float4)(-1.0f, -1.5f, 0.5f, -1.0f) / 8.0f;
    const float4 kB = (float4)(2.0f, 0.0f, 0.0f, 4.0f) / 8.0f;
    const float4 kC = (float4)(4.0f, 6.0f, 5.0f, 5.0f) / 8.0f;
    const float4 kD = (float4)(0.0f, 2.0f, -1.0f, -1.0f) / 8.0f;

    // The four filter responses: x = cross, y = checker, z = theta, w = phi.
    // Which of them yields which color depends on the pixel's tile position.
    float4 pattern = kC * C;
    pattern += (float4)(0.0f, kD.y, kD.z, kD.z) * D;
    pattern += (float4)(kA.x, kA.y, kA.z, kA.x) * A;
    pattern += (float4)(kA.x, kA.y, kA.x, kA.z) * E;
    pattern += (float4)(kB.x, 0.0f, 0.0f, kB.w) * B;
    pattern += (float4)(kB.x, 0.0f, kB.w, 0.0f) * F;

    float3 color;
    if (rowParity == 0)
        color = columnParity == 0 ? (float3)(C, pattern.x, pattern.y)   // red pixel
                                  : (float3)(pattern.z, C, pattern.w);  // green pixel in a red row
    else
        color = columnParity == 0 ? (float3)(pattern.w, C, pattern.z)   // green pixel in a blue row
                                  : (float3)(pattern.y, pattern.x, C);  // blue pixel

    const int out = (y * width + x) * 3;
    rgb[out] = convert_uchar_sat_rte(color.x * gainRed);
    rgb[out + 1] = convert_uchar_sat_rte(color.y * gainGreen);
    rgb[out + 2] = convert_uchar_sat_rte(color.z * gainBlue);
#undef FETCH
}
)CLC";

} // namespace sample
