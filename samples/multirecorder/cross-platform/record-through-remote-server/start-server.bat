@echo off
rem Starts a MultiRecorder camera server for the cameras connected to this
rem computer. Pass the address to listen at; the default localhost:8090 accepts
rem only local clients, so use this computer's IP address (for example
rem 192.168.1.10:8090) when record-through-remote-server runs elsewhere.
rem MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
rem full path of MultiRecorder.exe to use another copy. Press Ctrl+C to stop.
setlocal
cd /d "%~dp0"
if not defined MULTIRECORDER set "MULTIRECORDER=%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe"
if not exist "%MULTIRECORDER%" (
    echo Error: MultiRecorder not found at "%MULTIRECORDER%". Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the full path of MultiRecorder.exe. 1>&2
    exit /b 1
)
set "ADDRESS=%~1"
if not defined ADDRESS set "ADDRESS=localhost:8090"
"%MULTIRECORDER%" --start-server "%ADDRESS%"
exit /b %ERRORLEVEL%
