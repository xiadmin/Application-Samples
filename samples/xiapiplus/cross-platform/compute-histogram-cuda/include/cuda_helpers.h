#pragma once

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

namespace sample
{

// Throws std::runtime_error with the CUDA error name and description when
// status is not cudaSuccess.
void checkCuda(cudaError_t status, const char* p_call);

struct CudaFree
{
    void operator()(void* p_memory) const;
};

// Device memory released with cudaFree() when the owner goes out of scope.
template <typename T>
using DeviceBuffer = std::unique_ptr<T, CudaFree>;

template <typename T>
DeviceBuffer<T> allocateDeviceBuffer(std::size_t count)
{
    void* p_memory = nullptr;
    checkCuda(cudaMalloc(&p_memory, count * sizeof(T)), "cudaMalloc");
    return DeviceBuffer<T>(static_cast<T*>(p_memory));
}

} // namespace sample
