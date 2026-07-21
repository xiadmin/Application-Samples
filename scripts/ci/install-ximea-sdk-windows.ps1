$ErrorActionPreference = "Stop"

$sdkUrl = "https://www.ximea.com/getattachment/b7b2379e-d9d1-4f79-8c87-d2055c473b8c/XIMEA_Windows_SP_Stable.exe"
$sdkSha256 = "3573c6572ca7ad02ec9e381c3d8df90af486008eeaadee3f41407a533b91f6ed"
$downloadDir = Join-Path $env:RUNNER_TEMP "ximea-download"
$sdkRoot = Join-Path $env:RUNNER_TEMP "ximea-sdk"
$installer = Join-Path $downloadDir "XIMEA_Windows_SP_V4.32.00.exe"
$sevenZip = Join-Path $env:ProgramFiles "7-Zip\7z.exe"

if ([string]::IsNullOrWhiteSpace($env:RUNNER_TEMP)) {
    throw "RUNNER_TEMP must be set by GitHub Actions"
}
if ([string]::IsNullOrWhiteSpace($env:GITHUB_ENV)) {
    throw "GITHUB_ENV must be set by GitHub Actions"
}
if (-not (Test-Path -PathType Leaf $sevenZip)) {
    throw "7-Zip is required at $sevenZip"
}

New-Item -ItemType Directory -Force -Path $downloadDir, $sdkRoot | Out-Null
Invoke-WebRequest -Uri $sdkUrl -OutFile $installer

$actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $installer).Hash.ToLowerInvariant()
if ($actualSha256 -ne $sdkSha256) {
    throw "XIMEA SDK checksum mismatch: expected $sdkSha256, got $actualSha256"
}

$signature = Get-AuthenticodeSignature -FilePath $installer
if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "XIMEA") {
    throw "XIMEA SDK Authenticode signature is not valid"
}

& $sevenZip x -y $installer "-o$sdkRoot"
if ($LASTEXITCODE -ne 0) {
    throw "7-Zip failed to unpack the XIMEA SDK with exit code $LASTEXITCODE"
}

$env:XIMEA_SP_PATH = $sdkRoot
$env:PYTHONPATH = "$sdkRoot\API\Python\v3"
$env:PATH = "$sdkRoot\API\xiAPI;$env:PATH"

"XIMEA_SP_PATH=$env:XIMEA_SP_PATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"PYTHONPATH=$env:PYTHONPATH" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"$sdkRoot\API\xiAPI" | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append

Write-Host "Installed XIMEA SDK LTS V4.32.00 for Windows x64 at $env:XIMEA_SP_PATH"
