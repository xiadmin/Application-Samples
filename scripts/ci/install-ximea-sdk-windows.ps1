$ErrorActionPreference = "Stop"

$sdkUrl = "https://www.ximea.com/getattachment/23c0b9e6-5c24-4d27-9a6e-b377d9390c3d/XIMEA_Windows_SP_Beta.exe"
$downloadDir = Join-Path $env:RUNNER_TEMP "ximea-download"
$installer = Join-Path $downloadDir "XIMEA_Windows_SP_Beta_latest.exe"

if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    throw "RUNNER_TEMP must be set by GitHub Actions"
}
if ([string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    throw "GITHUB_ENV must be set by GitHub Actions"
}

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
Invoke-WebRequest -Uri $sdkUrl -OutFile $installer

$signature = Get-AuthenticodeSignature -FilePath $installer
if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "XIMEA") {
    throw "XIMEA SDK Authenticode signature is not valid"
}

$installerProcess = Start-Process -FilePath $installer -ArgumentList "/S" -Wait -PassThru
if ($installerProcess.ExitCode -ne 0) {
    throw "XIMEA SDK installer failed with exit code $($installerProcess.ExitCode)"
}

$ximeaSpPath = [Environment]::GetEnvironmentVariable("XIMEA_SP_PATH", "Machine")
if ([string]::IsNullOrWhiteSpace($ximeaSpPath)) {
    $ximeaSpPath = [Environment]::GetEnvironmentVariable("XIMEA_SP_PATH", "User")
}
if ([string]::IsNullOrWhiteSpace($ximeaSpPath)) {
    throw "XIMEA SDK installer completed but did not set XIMEA_SP_PATH"
}

$env:XIMEA_SP_PATH = $ximeaSpPath
$env:PYTHONPATH = "$ximeaSpPath\API\Python\v3"
$env:PATH = "$ximeaSpPath\API\xiAPI;$env:PATH"

"XIMEA_SP_PATH=$env:XIMEA_SP_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"PYTHONPATH=$env:PYTHONPATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"$ximeaSpPath\API\xiAPI" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

Write-Host "Installed latest XIMEA SDK beta for Windows x64 at $env:XIMEA_SP_PATH"
