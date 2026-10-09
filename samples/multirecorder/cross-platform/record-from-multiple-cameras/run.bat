@echo off
rem Records 10 images from each of three cameras and exports them as TIFF files.
rem MultiRecorder comes from the XIMEA Software Package; set MULTIRECORDER to the
rem full path of MultiRecorder.exe to use another copy.
setlocal
cd /d "%~dp0"
if not defined MULTIRECORDER set "MULTIRECORDER=%XIMEA_SP_PATH%\Examples\MultiRecorder\MultiRecorder.exe"
if not exist "%MULTIRECORDER%" (
    echo Error: MultiRecorder not found at "%MULTIRECORDER%". Install the XIMEA Software Package with its examples, or set MULTIRECORDER to the full path of MultiRecorder.exe. 1>&2
    exit /b 1
)
"%MULTIRECORDER%" --run setup.xml
exit /b %ERRORLEVEL%
