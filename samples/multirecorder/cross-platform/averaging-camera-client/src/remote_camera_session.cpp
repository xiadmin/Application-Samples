#include <iostream>
#include <stdexcept>

#include "remote_camera_session.h"

namespace sample
{

void checkXi(int status, const char* p_call)
{
    if (status != XI_OK)
        throw std::runtime_error(std::string(p_call) + " returned xiAPI error " + std::to_string(status));
}

RemoteCameraSession::RemoteCameraSession(CxRemoteCameraClient* p_camera) : m_camera(p_camera)
{
    if (!m_camera)
        throw std::invalid_argument("RemoteCameraSession needs an opened camera");
}

RemoteCameraSession::~RemoteCameraSession()
{
    if (m_isAcquiring)
    {
        const int status = m_camera->stopAcquisition();
        if (status != XI_OK)
            std::cerr << "Warning: stopping the acquisition on the server returned xiAPI error " << status << "\n";
    }
    const int status = m_camera->closeDevice();
    if (status != XI_OK)
        std::cerr << "Warning: closing the camera on the server returned xiAPI error " << status << "\n";
}

void RemoteCameraSession::startAcquisition()
{
    checkXi(m_camera->startAcquisition(), "startAcquisition");
    m_isAcquiring = true;
}

void RemoteCameraSession::stopAcquisition()
{
    m_isAcquiring = false;
    checkXi(m_camera->stopAcquisition(), "stopAcquisition");
}

} // namespace sample
