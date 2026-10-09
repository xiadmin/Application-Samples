#include "cuda_helpers.h"

#include <stdexcept>
#include <string>

namespace sample
{

void checkCuda(cudaError_t status, const char* p_call)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(p_call) + " failed with CUDA error " + cudaGetErrorName(status) + ": " +
                                 cudaGetErrorString(status));
    }
}

void CudaFree::operator()(void* p_memory) const
{
    cudaFree(p_memory);
}

} // namespace sample
