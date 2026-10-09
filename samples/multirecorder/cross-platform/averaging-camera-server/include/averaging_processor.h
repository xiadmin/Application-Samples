#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <UserImageProcessor.h>

class TiXmlElement;

namespace sample
{

// Averages every N consecutive camera frames on the server, so that only the
// averaged images travel to the client. The client sets N with a user image
// processing request that carries an averaging="N" attribute; N = 1 passes
// every frame through unchanged. Works on XI_MONO8, XI_RAW8 and XI_RGB24.
class AveragingProcessor : public CxUserImageProcessor
{
public:
    bool command(TiXmlElement* p_request, std::string& response, std::string& responseContentType) override;
    bool processImage(XI_IMG* p_image, std::string& response, std::string& responseContentType) override;

private:
    void restartIfLayoutChanged(const XI_IMG& image, std::uint32_t componentCount);
    void addToSum(const XI_IMG& image, std::uint32_t rowBytes, std::uint32_t stride);
    void makeAverageResponse(const XI_IMG& image, std::uint32_t rowBytes, std::uint32_t stride, std::string& response,
                             std::string& responseContentType);

    // The server handles control commands and image requests on different
    // threads, so every member below is guarded by this mutex.
    std::mutex m_mutex;
    int m_averagingCount = 1;
    int m_summedCount = 0;
    std::vector<std::uint32_t> m_sum;
    std::vector<std::uint8_t> m_average;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    std::uint32_t m_paddingX = 0;
    XI_IMG_FORMAT m_format = XI_MONO8;
};

} // namespace sample
