@echo off
rem Starts a camera server on this computer, records 10 images through it, then stops the server.
rem MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
rem full path of MultiRecorder.exe to use another copy.
setlocal
cd /d "%~dp0"
if not defined MULTIRECORDER set "MULTIRECORDER=%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe"
if not exist "%MULTIRECORDER%" (
    echo Error: MultiRecorder not found at "%MULTIRECORDER%". Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the full path of MultiRecorder.exe. 1>&2
    exit /b 1
)
rem The camera server runs in its own minimized window; the window title
rem identifies it for the shutdown below.
start "multirecorder-sample-server" /min "%MULTIRECORDER%" --start-server localhost:8090 --log server-log.txt
rem Give the server a moment to start listening.
powershell -NoProfile -Command "Start-Sleep -Seconds 1"
"%MULTIRECORDER%" --run setup.xml
set "RESULT=%ERRORLEVEL%"
taskkill /fi "WINDOWTITLE eq multirecorder-sample-server" /f >nul
exit /b %RESULT%
