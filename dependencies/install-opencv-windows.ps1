<#
.SYNOPSIS
    Installs a pinned, SHA-256-checked OpenCV release on Windows and configures
    OpenCV_DIR and PATH so that the OpenCV samples can be built and run.

.DESCRIPTION
    - Downloads the official OpenCV release installer
      https://github.com/opencv/opencv/releases/download/<version>/opencv-<version>-windows.exe
      into <repository>\dependencies\cache\downloads (shared with the CMake archive cache) and
      accepts it only if its SHA256 equals -OpenCvSha256. A cached copy is reused
      only when it matches as well. Nothing is run before the hash matches.
    - Unpacks it (a 7-Zip self-extracting archive) into <InstallRoot>\opencv. An
      OpenCV of another version found there is replaced by the pinned one.
    - Sets OpenCV_DIR to the OpenCV build tree and puts the OpenCV DLL folder on PATH.
      In GitHub Actions the values go to GITHUB_ENV and GITHUB_PATH; otherwise they
      are persisted for the chosen -Scope and set in the current session.

    No Chocolatey and, by default, no Administrator rights are needed:
        powershell -ExecutionPolicy Bypass -File dependencies\install-opencv-windows.ps1
    Administrator rights are needed only for -Scope Machine or an -InstallRoot
    the current user cannot write to; add -SelfElevate to have the script ask for
    them through UAC in that case. CMake runs this script only when OpenCV is
    missing and the environment variable AUTOMATICALLY_DOWNLOAD_DEPENDENCIES is
    set to 1, ON, TRUE or YES.

.PARAMETER OpenCvVersion
    OpenCV release to install. This default is the single place where the Windows
    OpenCV version is pinned; CI uses this script as well. It must match the
    OpenCV version of the opencv-python pin in dependencies/python-constraints.txt.

.PARAMETER OpenCvSha256
    SHA-256 of opencv-<OpenCvVersion>-windows.exe. OpenCV publishes no signature
    or checksum file for it; this value was compared against the digest GitHub
    reports for the release asset (API: repos/opencv/opencv/releases/tags/<version>)
    and against a fresh download. Change it together with -OpenCvVersion.

.PARAMETER Scope
    Where OpenCV_DIR and PATH are persisted: User (default) or Machine.
    Machine needs Administrator rights.

.PARAMETER InstallRoot
    Folder the installer unpacks into. OpenCV lands in <InstallRoot>\opencv.
    cmake/CMakeLists.txt looks for OpenCV under the default C:\tools.

.PARAMETER DownloadDir
    Where the installer is cached. Default: <repository>\dependencies\cache\downloads.

.PARAMETER SelfElevate
    When Administrator rights are needed but missing, start this script again as
    Administrator (UAC prompt), wait for it and show its output, instead of
    stopping with an error. Has no effect when no elevation is needed.

.PARAMETER TranscriptPath
    Internal: where the elevated copy started by -SelfElevate writes its output.
#>
[CmdletBinding()]
param(
    [string] $OpenCvVersion = '4.14.0',
    [string] $OpenCvSha256 = '5f266a8b73bed535962d7e861a6457e32a0dd5f463ad0a7cf8707a135469be63',
    [ValidateSet('User', 'Machine')]
    [string] $Scope = 'User',
    [string] $InstallRoot = 'C:\tools',
    [string] $DownloadDir,
    [switch] $SelfElevate,
    [string] $TranscriptPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

trap {
    # In the elevated copy, record the error in the transcript before exiting.
    if ($TranscriptPath) {
        Write-Host "ERROR: $_"
        try { Stop-Transcript | Out-Null } catch { }
    }
    break
}

function Test-PathListContains {
    param([string] $PathList, [string] $Directory)
    $wanted = $Directory.TrimEnd('\')
    foreach ($entry in ($PathList -split ';')) {
        if ($entry -and ([Environment]::ExpandEnvironmentVariables($entry).TrimEnd('\') -ieq $wanted)) {
            return $true
        }
    }
    return $false
}

# Creates the folder if needed and reports whether the current user can write to it.
function Test-DirectoryWritable {
    param([string] $Directory)
    try {
        New-Item -ItemType Directory -Force -Path $Directory | Out-Null
        $probe = Join-Path $Directory ".write-test-$PID"
        [IO.File]::WriteAllText($probe, '')
        Remove-Item -LiteralPath $probe
        return $true
    }
    catch {
        return $false
    }
}

# OpenCV_VERSION from the OpenCV CMake package under $BuildDir, or $null.
function Get-InstalledOpenCvVersion {
    param([string] $BuildDir)
    $versionFile = Join-Path $BuildDir 'OpenCVConfig-version.cmake'
    if (-not (Test-Path -LiteralPath $versionFile)) { return $null }
    foreach ($line in (Get-Content -LiteralPath $versionFile)) {
        if ($line -match '^\s*set\(\s*OpenCV_VERSION\s+"?([0-9][0-9.]*)"?\s*\)') {
            return $Matches[1]
        }
    }
    return $null
}

function Get-Sha256 {
    param([string] $Path)
    # Not Get-FileHash: in Windows PowerShell 5.1 it lives in a script module
    # that is auto-loaded through PSModulePath. Started from PowerShell 7 (for
    # example by CMake inside a pwsh CI step), powershell.exe inherits pwsh's
    # PSModulePath, finds the PowerShell 7 copy of that module, and
    # Get-FileHash is "not recognized". The .NET class works in every host.
    $fullPath = (Resolve-Path -LiteralPath $Path).ProviderPath
    $stream = [System.IO.File]::OpenRead($fullPath)
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hash = $sha256.ComputeHash($stream)
        } finally {
            $sha256.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

if ($OpenCvSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw "-OpenCvSha256 must be the 64 hexadecimal digits of the installer's SHA-256 hash."
}
$OpenCvSha256 = $OpenCvSha256.ToLowerInvariant()
if (-not $DownloadDir) {
    $DownloadDir = Join-Path $PSScriptRoot 'cache\downloads'
}
$DownloadDir = [IO.Path]::GetFullPath($DownloadDir)
$InstallRoot = [IO.Path]::GetFullPath($InstallRoot)
$installerName = "opencv-$OpenCvVersion-windows.exe"
$installerUrl = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/$installerName"
$openCvRoot = Join-Path $InstallRoot 'opencv'
$openCvBuild = Join-Path $openCvRoot 'build'

# 1. Elevation, only when it is actually required: a Machine-scope variable, or
#    a folder the current user cannot write to. GitHub Actions takes the variables
#    through GITHUB_ENV, so -Scope does not matter there.
$elevationReasons = @()
if ($Scope -eq 'Machine' -and -not $env:GITHUB_ENV) { $elevationReasons += '-Scope Machine' }
if (-not (Test-DirectoryWritable $InstallRoot)) { $elevationReasons += "writing to $InstallRoot" }
if (-not (Test-DirectoryWritable $DownloadDir)) { $elevationReasons += "writing to $DownloadDir" }

$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
$isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if ($elevationReasons.Count -gt 0 -and -not $isAdministrator) {
    $reason = $elevationReasons -join ' and '
    if (-not $SelfElevate) {
        throw "$reason needs Administrator rights. Run PowerShell as Administrator and start the script again, add -SelfElevate, or choose -Scope User and an -InstallRoot you can write to."
    }

    # The elevated copy runs in its own window, so it writes a transcript that
    # is shown here once it ends. With a separate administrator account, User
    # scope variables land in that account; hence elevation only when required.
    $log = Join-Path ([IO.Path]::GetTempPath()) "install-opencv-windows-$PID.log"
    $childArgs = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
        '-OpenCvVersion', $OpenCvVersion, '-OpenCvSha256', $OpenCvSha256, '-Scope', $Scope,
        '-InstallRoot', "`"$InstallRoot`"", '-DownloadDir', "`"$DownloadDir`"",
        '-TranscriptPath', "`"$log`"")
    Write-Host "Asking for Administrator rights ($reason)..."
    try {
        $child = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList $childArgs
    }
    catch {
        throw "Administrator rights were not granted: $($_.Exception.Message)"
    }
    if (Test-Path $log) {
        Get-Content $log | Write-Host
        Remove-Item $log -ErrorAction SilentlyContinue
    }
    if ($child.ExitCode -ne 0) {
        throw "The elevated OpenCV install failed with exit code $($child.ExitCode)."
    }
    if (-not $env:GITHUB_ENV) {
        # The elevated copy set the persistent variables; mirror them here.
        $env:OpenCV_DIR = [Environment]::GetEnvironmentVariable('OpenCV_DIR', $Scope)
    }
    return
}

if ($TranscriptPath) {
    Start-Transcript -Path $TranscriptPath -Force | Out-Null
}

# 2. Installed version: keep the pinned one, replace any other.
$installedVersion = Get-InstalledOpenCvVersion $openCvBuild
$needsInstall = $true
if ($installedVersion -eq $OpenCvVersion) {
    Write-Host "OpenCV $OpenCvVersion is already installed in $openCvRoot."
    $needsInstall = $false
}
elseif ($installedVersion) {
    Write-Warning "OpenCV $installedVersion is installed in $openCvRoot; replacing it with the pinned version $OpenCvVersion."
}
elseif (Test-Path -LiteralPath $openCvRoot) {
    # Not an OpenCV build tree this script can identify; do not delete it.
    throw "$openCvRoot exists but holds no OpenCV CMake package (build\OpenCVConfig-version.cmake). Remove it, or choose another -InstallRoot."
}

# An earlier version of this script installed OpenCV through Chocolatey. Only
# point that out; this script never runs choco.
$chocolateyRoot = if ($env:ChocolateyInstall) { $env:ChocolateyInstall } else { Join-Path $env:ProgramData 'chocolatey' }
if (Test-Path -LiteralPath (Join-Path $chocolateyRoot 'lib\opencv')) {
    Write-Host "Note: a Chocolatey 'opencv' package is also registered. To remove it, run 'choco uninstall opencv' from an Administrator PowerShell."
}

if ($needsInstall) {
    # 3. Installer: cached or downloaded, accepted only with the pinned SHA256.
    $installer = Join-Path $DownloadDir $installerName
    if (Test-Path -LiteralPath $installer) {
        $cachedSha256 = Get-Sha256 $installer
        if ($cachedSha256 -ne $OpenCvSha256) {
            throw "The cached installer $installer has SHA256 $cachedSha256, but the script expects $OpenCvSha256. The file is damaged or was tampered with. Delete it to download it again."
        }
        Write-Host "Reusing verified cached $installerName."
    }
    else {
        $partial = "$installer.partial"
        Write-Host "Downloading $installerUrl"
        [Net.ServicePointManager]::SecurityProtocol =
            [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        # The progress bar slows Invoke-WebRequest down many times in Windows PowerShell.
        $savedProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $partial
        }
        finally {
            $ProgressPreference = $savedProgress
        }
        $downloadedSha256 = Get-Sha256 $partial
        if ($downloadedSha256 -ne $OpenCvSha256) {
            Remove-Item -LiteralPath $partial -ErrorAction SilentlyContinue
            throw "The downloaded $installerName has SHA256 $downloadedSha256, but the script expects $OpenCvSha256. It was deleted and not run."
        }
        Move-Item -LiteralPath $partial -Destination $installer -Force
        Write-Host "Downloaded and verified $installerName."
    }

    # 4. Unpack. The installer is a 7-Zip self-extracting archive: -o sets the
    #    target folder, -y answers every prompt. It is a GUI program, so
    #    Start-Process -Wait is needed to wait for it and read its exit code.
    if ($installedVersion) {
        Remove-Item -LiteralPath $openCvRoot -Recurse -Force
    }
    Write-Host "Unpacking OpenCV $OpenCvVersion into $openCvRoot."
    $extract = Start-Process -FilePath $installer -ArgumentList "-o`"$InstallRoot`"", '-y' -Wait -PassThru
    if ($extract.ExitCode -ne 0) {
        # $openCvRoot did not exist before this step (see step 2), so the
        # partial tree is this script's own and can go.
        Remove-Item -LiteralPath $openCvRoot -Recurse -Force -ErrorAction SilentlyContinue
        throw "Unpacking $installerName into $InstallRoot failed with exit code $($extract.ExitCode). The archive holds paths of about 155 characters below <InstallRoot>, so a long -InstallRoot exceeds the Windows 260-character path limit; choose a short one such as C:\tools."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $openCvBuild 'OpenCVConfig.cmake'))) {
        throw "Unpacking finished, but $openCvBuild\OpenCVConfig.cmake is missing."
    }
    $unpackedVersion = Get-InstalledOpenCvVersion $openCvBuild
    if ($unpackedVersion -ne $OpenCvVersion) {
        throw "The unpacked OpenCV reports version '$unpackedVersion', expected $OpenCvVersion."
    }
}

# 5. Newest Visual C++ DLL folder in the build tree.
$openCvBin = Get-ChildItem (Join-Path $openCvBuild 'x64\vc*\bin') -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $openCvBin) {
    throw "OpenCV binaries not found under $openCvBuild\x64."
}
$openCvBinDir = $openCvBin.FullName

# 6. Environment.
if ($env:GITHUB_ENV) {
    "OpenCV_DIR=$openCvBuild" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    $openCvBinDir | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
    Write-Host 'Wrote OpenCV_DIR to GITHUB_ENV and the OpenCV bin folder to GITHUB_PATH.'
}
else {
    # PATH is edited in the registry directly so that unexpanded entries such as
    # %SystemRoot% and the REG_EXPAND_SZ type survive the update.
    if ($Scope -eq 'Machine') {
        $envKey = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey(
            'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', $true)
    }
    else {
        $envKey = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
    }
    try {
        $rawPath = [string]$envKey.GetValue('Path', '',
            [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (Test-PathListContains $rawPath $openCvBinDir) {
            Write-Host "$Scope PATH already contains $openCvBinDir."
        }
        else {
            $newPath = if ($rawPath) { "$openCvBinDir;$rawPath" } else { $openCvBinDir }
            $envKey.SetValue('Path', $newPath, [Microsoft.Win32.RegistryValueKind]::ExpandString)
            Write-Host "Added $openCvBinDir to the $Scope PATH."
        }
    }
    finally {
        $envKey.Close()
    }

    # SetEnvironmentVariable also tells running programs that the environment changed.
    [Environment]::SetEnvironmentVariable('OpenCV_DIR', $openCvBuild, $Scope)

    $env:OpenCV_DIR = $openCvBuild
    if (-not (Test-PathListContains $env:Path $openCvBinDir)) {
        $env:Path = "$openCvBinDir;$env:Path"
    }
    Write-Host 'Restart open terminals and IDEs so that they see the new OpenCV_DIR and PATH.'
}

# 7. Summary.
Write-Host ''
Write-Host "OpenCV version : $OpenCvVersion"
Write-Host "Installer      : $installerUrl"
Write-Host "SHA256         : $OpenCvSha256"
Write-Host "OpenCV_DIR     : $openCvBuild"
Write-Host "OpenCV bin     : $openCvBinDir"

if ($TranscriptPath) {
    Stop-Transcript | Out-Null
}
