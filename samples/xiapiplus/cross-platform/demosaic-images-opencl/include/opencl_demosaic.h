#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace sample
{

// Thrown for every OpenCL call that does not return CL_SUCCESS.
class OpenClError : public std::runtime_error
{
public:
    OpenClError(const std::string& call, cl_int status);

    cl_int status() const { return m_status; }

private:
    cl_int m_status;
};

void checkCl(cl_int status, const char* p_call);

// Releases an OpenCL object; std::unique_ptr picks the overload by type.
struct ClReleaser
{
    void operator()(cl_context p_context) const;
    void operator()(cl_command_queue p_queue) const;
    void operator()(cl_program p_program) const;
    void operator()(cl_kernel p_kernel) const;
    void operator()(cl_mem p_memory) const;
    void operator()(cl_event p_event) const;
};

template <typename T>
using ClHandle = std::unique_ptr<std::remove_pointer_t<T>, ClReleaser>;

// Gains applied to the demosaiced channels, normally the camera's white
// balance coefficients.
struct ChannelGains
{
    float red = 1.0f;
    float green = 1.0f;
    float blue = 1.0f;
};

struct DemosaicTiming
{
    double kernelMs = 0.0; // kernel execution measured by the OpenCL device
    double totalMs = 0.0;  // upload, kernel and download measured on the host
};

// Demosaics RAW8 Bayer frames of a fixed size on the first OpenCL GPU, or on
// any OpenCL device when there is no GPU.
class OpenClDemosaic
{
public:
    OpenClDemosaic(int width, int height);

    // Converts one frame into interleaved 8-bit RGB (width * height * 3 bytes).
    // rawStride is the distance between rows in bytes; (redX, redY) is the
    // position of the red pixel in the frame's top-left 2x2 tile.
    DemosaicTiming process(const std::uint8_t* p_raw, int rawStride, int redX, int redY, const ChannelGains& gains,
                           std::vector<std::uint8_t>& rgb);

    const std::string& deviceName() const { return m_deviceName; }

private:
    int m_width;
    int m_height;
    std::string m_deviceName;
    ClHandle<cl_context> m_context;
    ClHandle<cl_command_queue> m_queue;
    ClHandle<cl_program> m_program;
    ClHandle<cl_kernel> m_kernel;
    ClHandle<cl_mem> m_rawBuffer;
    std::size_t m_rawBufferBytes = 0;
    ClHandle<cl_mem> m_rgbBuffer;
};

} // namespace sample
