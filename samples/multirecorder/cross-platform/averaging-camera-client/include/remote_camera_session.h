#pragma once

#include <memory>
#include <string>

#include <RemoteCameraClient.h>

namespace sample
{

// Throws std::runtime_error when a CxRemoteCameraClient call does not return XI_OK.
void checkXi(int status, const char* p_call);

// Owns a camera opened on a camera server. The destructor stops a running
// acquisition and closes the camera on the server, also when an exception
// unwinds the stack; failures there are reported on stderr.
class RemoteCameraSession
{
public:
    explicit RemoteCameraSession(CxRemoteCameraClient* p_camera);
    ~RemoteCameraSession();

    RemoteCameraSession(const RemoteCameraSession&) = delete;
    RemoteCameraSession& operator=(const RemoteCameraSession&) = delete;

    CxRemoteCameraClient& camera() { return *m_camera; }
    void startAcquisition();
    void stopAcquisition();

private:
    std::unique_ptr<CxRemoteCameraClient> m_camera;
    bool m_isAcquiring = false;
};

} // namespace sample
