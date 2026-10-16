#include <algorithm>

#include <cuda_runtime.h>

#include "cuda_helpers.h"
#include "histogram64.h"

namespace sample
{

namespace
{

constexpr unsigned int threadsPerBlock = 256;
// Enough resident blocks to keep every multiprocessor busy; each thread then
// loops over the frame with a grid-sized stride.
constexpr unsigned int blocksPerMultiprocessor = 8;

// Every block counts into its own histogram in shared memory, where atomic
// additions are cheap, and adds it to the global result once per bin.
__global__ void histogram64Kernel(const unsigned char* p_data, std::size_t byteCount, unsigned int* p_histogram)
{
    __shared__ unsigned int blockHistogram[histogramBinCount];
    for (unsigned int bin = threadIdx.x; bin < histogramBinCount; bin += blockDim.x)
        blockHistogram[bin] = 0;
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < byteCount; i += stride)
    {
        // 256 grey levels map onto 64 bins, so the lowest two bits are dropped.
        atomicAdd(&blockHistogram[p_data[i] >> 2], 1u);
    }
    __syncthreads();

    for (unsigned int bin = threadIdx.x; bin < histogramBinCount; bin += blockDim.x)
        atomicAdd(&p_histogram[bin], blockHistogram[bin]);
}

} // namespace

void computeHistogram64(const unsigned char* p_deviceData, std::size_t byteCount, unsigned int* p_deviceHistogram)
{
    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    int multiprocessorCount = 0;
    checkCuda(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device),
              "cudaDeviceGetAttribute(MultiProcessorCount)");

    const std::size_t neededBlocks = (byteCount + threadsPerBlock - 1) / threadsPerBlock;
    const std::size_t maxBlocks = static_cast<std::size_t>(multiprocessorCount) * blocksPerMultiprocessor;
    const auto blockCount = static_cast<unsigned int>(std::max<std::size_t>(1, std::min(neededBlocks, maxBlocks)));

    checkCuda(cudaMemset(p_deviceHistogram, 0, histogramBinCount * sizeof(unsigned int)), "cudaMemset");
    histogram64Kernel<<<blockCount, threadsPerBlock>>>(p_deviceData, byteCount, p_deviceHistogram);
    checkCuda(cudaGetLastError(), "histogram64Kernel launch");
}

} // namespace sample
