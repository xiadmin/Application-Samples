#include <algorithm>
#include <iostream>

#include <tinyxml.h>

#include <CameraControl.h>

#include "averaging_processor.h"

namespace sample
{

namespace
{

// Bytes per pixel of the 8-bit formats the processor averages; 0 for others.
std::uint32_t componentCountFor(XI_IMG_FORMAT format)
{
    switch (format)
    {
    case XI_MONO8:
    case XI_RAW8:
        return 1;
    case XI_RGB24:
        return 3;
    default:
        return 0;
    }
}

void setTextResponse(const std::string& text, std::string& response, std::string& responseContentType)
{
    response = text;
    responseContentType = "text/plain";
}

} // namespace

bool AveragingProcessor::command(TiXmlElement* p_request, std::string& response, std::string& responseContentType)
{
    int averagingCount = 0;
    if (p_request == nullptr || p_request->QueryIntAttribute("averaging", &averagingCount) != TIXML_SUCCESS ||
        averagingCount < 1)
    {
        setTextResponse("Unsupported command; expected averaging=\"<frames>\" with at least 1 frame", response,
                        responseContentType);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_averagingCount = averagingCount;
        m_summedCount = 0;
    }
    std::cout << "Averaging set to " << averagingCount << " frames" << std::endl;
    setTextResponse("OK", response, responseContentType);
    return true;
}

bool AveragingProcessor::processImage(XI_IMG* p_image, std::string& response, std::string& responseContentType)
{
    if (p_image == nullptr || p_image->bp == nullptr)
    {
        setTextResponse("No image", response, responseContentType);
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_averagingCount <= 1)
    {
        CxCameraControl::makeXiImgResponse(p_image, response, responseContentType);
        return true;
    }

    const std::uint32_t componentCount = componentCountFor(p_image->frm);
    if (componentCount == 0)
    {
        setTextResponse("Averaging supports XI_MONO8, XI_RAW8 and XI_RGB24 only", response, responseContentType);
        return false;
    }
    const std::uint32_t rowBytes = p_image->width * componentCount;
    const std::uint32_t stride = rowBytes + p_image->padding_x;
    if (p_image->bp_size < stride * p_image->height)
    {
        setTextResponse("The image buffer is smaller than its size and format require", response, responseContentType);
        return false;
    }

    restartIfLayoutChanged(*p_image, componentCount);
    addToSum(*p_image, rowBytes, stride);
    if (m_summedCount < m_averagingCount)
    {
        setTextResponse("Averaging frame " + std::to_string(m_summedCount) + "/" + std::to_string(m_averagingCount),
                        response, responseContentType);
        return true;
    }

    makeAverageResponse(*p_image, rowBytes, stride, response, responseContentType);
    m_summedCount = 0;
    return true;
}

void AveragingProcessor::restartIfLayoutChanged(const XI_IMG& image, std::uint32_t componentCount)
{
    // A different size or format cannot be added to the frames summed so far.
    if (image.width == m_width && image.height == m_height && image.padding_x == m_paddingX && image.frm == m_format &&
        !m_sum.empty())
        return;

    m_width = image.width;
    m_height = image.height;
    m_paddingX = image.padding_x;
    m_format = image.frm;
    m_sum.assign(static_cast<std::size_t>(m_width) * componentCount * m_height, 0);
    m_summedCount = 0;
}

void AveragingProcessor::addToSum(const XI_IMG& image, std::uint32_t rowBytes, std::uint32_t stride)
{
    if (m_summedCount == 0)
        std::fill(m_sum.begin(), m_sum.end(), 0);

    const auto* p_pixels = static_cast<const std::uint8_t*>(image.bp);
    for (std::uint32_t row = 0; row < image.height; ++row)
    {
        const std::uint8_t* p_source = p_pixels + static_cast<std::size_t>(row) * stride;
        std::uint32_t* p_sumRow = m_sum.data() + static_cast<std::size_t>(row) * rowBytes;
        for (std::uint32_t column = 0; column < rowBytes; ++column)
            p_sumRow[column] += p_source[column];
    }
    ++m_summedCount;
}

void AveragingProcessor::makeAverageResponse(const XI_IMG& image, std::uint32_t rowBytes, std::uint32_t stride,
                                             std::string& response, std::string& responseContentType)
{
    // The averaged image keeps the metadata and row layout of the last frame.
    m_average.assign(image.bp_size, 0);
    const auto count = static_cast<std::uint32_t>(m_summedCount);
    for (std::uint32_t row = 0; row < image.height; ++row)
    {
        const std::uint32_t* p_sumRow = m_sum.data() + static_cast<std::size_t>(row) * rowBytes;
        std::uint8_t* p_averageRow = m_average.data() + static_cast<std::size_t>(row) * stride;
        for (std::uint32_t column = 0; column < rowBytes; ++column)
            p_averageRow[column] = static_cast<std::uint8_t>((p_sumRow[column] + count / 2) / count);
    }

    XI_IMG averaged = image;
    averaged.bp = m_average.data();
    CxCameraControl::makeXiImgResponse(&averaged, response, responseContentType);
}

} // namespace sample
