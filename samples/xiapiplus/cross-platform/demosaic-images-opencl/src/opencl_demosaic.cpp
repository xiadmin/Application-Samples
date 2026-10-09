#include "opencl_demosaic.h"

#include <chrono>

#include "demosaic_kernel.h"

namespace sample
{

namespace
{

std::string deviceInfoString(cl_device_id device, cl_device_info info)
{
    std::size_t size = 0;
    checkCl(clGetDeviceInfo(device, info, 0, nullptr, &size), "clGetDeviceInfo");
    std::string text(size, '\0');
    checkCl(clGetDeviceInfo(device, info, size, text.data(), nullptr), "clGetDeviceInfo");
    while (!text.empty() && text.back() == '\0')
        text.pop_back();
    return text;
}

// Returns the first GPU of any platform, or else the first device of any type.
cl_device_id pickDevice()
{
    cl_uint platformCount = 0;
    const cl_int status = clGetPlatformIDs(0, nullptr, &platformCount);
    if (status != CL_SUCCESS || platformCount == 0)
        throw std::runtime_error("no OpenCL platform found; install the OpenCL runtime of the GPU driver");

    std::vector<cl_platform_id> platforms(platformCount);
    checkCl(clGetPlatformIDs(platformCount, platforms.data(), nullptr), "clGetPlatformIDs");

    for (cl_device_type type : {static_cast<cl_device_type>(CL_DEVICE_TYPE_GPU),
                                static_cast<cl_device_type>(CL_DEVICE_TYPE_ALL)})
    {
        for (cl_platform_id platform : platforms)
        {
            cl_device_id device = nullptr;
            cl_uint deviceCount = 0;
            if (clGetDeviceIDs(platform, type, 1, &device, &deviceCount) == CL_SUCCESS && deviceCount > 0)
                return device;
        }
    }
    throw std::runtime_error("no OpenCL device found; install the OpenCL runtime of the GPU driver");
}

double eventMilliseconds(cl_event event)
{
    cl_ulong startNs = 0;
    cl_ulong endNs = 0;
    checkCl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(startNs), &startNs, nullptr),
            "clGetEventProfilingInfo(START)");
    checkCl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(endNs), &endNs, nullptr),
            "clGetEventProfilingInfo(END)");
    return static_cast<double>(endNs - startNs) * 1e-6;
}

} // namespace

OpenClError::OpenClError(const std::string& call, cl_int status)
    : std::runtime_error(call + " failed with OpenCL error " + std::to_string(status)), m_status(status)
{
}

void checkCl(cl_int status, const char* p_call)
{
    if (status != CL_SUCCESS)
        throw OpenClError(p_call, status);
}

void ClReleaser::operator()(cl_context p_context) const
{
    clReleaseContext(p_context);
}

void ClReleaser::operator()(cl_command_queue p_queue) const
{
    clReleaseCommandQueue(p_queue);
}

void ClReleaser::operator()(cl_program p_program) const
{
    clReleaseProgram(p_program);
}

void ClReleaser::operator()(cl_kernel p_kernel) const
{
    clReleaseKernel(p_kernel);
}

void ClReleaser::operator()(cl_mem p_memory) const
{
    clReleaseMemObject(p_memory);
}

void ClReleaser::operator()(cl_event p_event) const
{
    clReleaseEvent(p_event);
}

OpenClDemosaic::OpenClDemosaic(int width, int height) : m_width(width), m_height(height)
{
    cl_device_id device = pickDevice();
    m_deviceName = deviceInfoString(device, CL_DEVICE_NAME);

    cl_int status = CL_SUCCESS;
    m_context.reset(clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status));
    checkCl(status, "clCreateContext");
    // Profiling lets the sample report the kernel time measured by the device.
    m_queue.reset(clCreateCommandQueue(m_context.get(), device, CL_QUEUE_PROFILING_ENABLE, &status));
    checkCl(status, "clCreateCommandQueue");

    // The C API takes a non-const array of source strings but does not modify it.
    const char* p_source = demosaicKernelSource;
    m_program.reset(clCreateProgramWithSource(m_context.get(), 1, &p_source, nullptr, &status));
    checkCl(status, "clCreateProgramWithSource");
    status = clBuildProgram(m_program.get(), 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS)
    {
        std::size_t logSize = 0;
        clGetProgramBuildInfo(m_program.get(), device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(m_program.get(), device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        throw std::runtime_error("building the demosaic kernel failed with OpenCL error " + std::to_string(status) +
                                 ":\n" + log);
    }
    m_kernel.reset(clCreateKernel(m_program.get(), "demosaic", &status));
    checkCl(status, "clCreateKernel");

    const std::size_t rgbBytes = static_cast<std::size_t>(width) * height * 3;
    m_rgbBuffer.reset(clCreateBuffer(m_context.get(), CL_MEM_WRITE_ONLY, rgbBytes, nullptr, &status));
    checkCl(status, "clCreateBuffer(rgb)");
}

DemosaicTiming OpenClDemosaic::process(const std::uint8_t* p_raw, int rawStride, int redX, int redY,
                                       const ChannelGains& gains, std::vector<std::uint8_t>& rgb)
{
    const auto start = std::chrono::steady_clock::now();

    // The row stride can include padding, so the input buffer follows it.
    const std::size_t rawBytes = static_cast<std::size_t>(rawStride) * m_height;
    if (rawBytes > m_rawBufferBytes)
    {
        cl_int status = CL_SUCCESS;
        m_rawBuffer.reset(clCreateBuffer(m_context.get(), CL_MEM_READ_ONLY, rawBytes, nullptr, &status));
        checkCl(status, "clCreateBuffer(raw)");
        m_rawBufferBytes = rawBytes;
    }
    checkCl(clEnqueueWriteBuffer(m_queue.get(), m_rawBuffer.get(), CL_FALSE, 0, rawBytes, p_raw, 0, nullptr, nullptr),
            "clEnqueueWriteBuffer");

    cl_mem rawBuffer = m_rawBuffer.get();
    cl_mem rgbBuffer = m_rgbBuffer.get();
    cl_kernel kernel = m_kernel.get();
    checkCl(clSetKernelArg(kernel, 0, sizeof(cl_mem), &rawBuffer), "clSetKernelArg(raw)");
    checkCl(clSetKernelArg(kernel, 1, sizeof(cl_int), &rawStride), "clSetKernelArg(rawStride)");
    checkCl(clSetKernelArg(kernel, 2, sizeof(cl_mem), &rgbBuffer), "clSetKernelArg(rgb)");
    checkCl(clSetKernelArg(kernel, 3, sizeof(cl_int), &redX), "clSetKernelArg(redX)");
    checkCl(clSetKernelArg(kernel, 4, sizeof(cl_int), &redY), "clSetKernelArg(redY)");
    checkCl(clSetKernelArg(kernel, 5, sizeof(cl_float), &gains.red), "clSetKernelArg(gainRed)");
    checkCl(clSetKernelArg(kernel, 6, sizeof(cl_float), &gains.green), "clSetKernelArg(gainGreen)");
    checkCl(clSetKernelArg(kernel, 7, sizeof(cl_float), &gains.blue), "clSetKernelArg(gainBlue)");

    // One work item per pixel; the runtime picks the work-group size.
    const std::size_t globalSize[2] = {static_cast<std::size_t>(m_width), static_cast<std::size_t>(m_height)};
    cl_event kernelEvent = nullptr;
    checkCl(clEnqueueNDRangeKernel(m_queue.get(), kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, &kernelEvent),
            "clEnqueueNDRangeKernel");
    const ClHandle<cl_event> kernelDone(kernelEvent);

    rgb.resize(static_cast<std::size_t>(m_width) * m_height * 3);
    checkCl(clEnqueueReadBuffer(m_queue.get(), rgbBuffer, CL_TRUE, 0, rgb.size(), rgb.data(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer");

    DemosaicTiming timing;
    timing.kernelMs = eventMilliseconds(kernelDone.get());
    timing.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return timing;
}

} // namespace sample
