<#
.SYNOPSIS
    Installs a pinned OpenCV version on Windows with Chocolatey and configures
    OpenCV_DIR and PATH so that the OpenCV samples can be built and run.

.DESCRIPTION
    - Installs Chocolatey first when it is missing (official chocolatey.org method).
    - Installs exactly the OpenCV version given by -OpenCvVersion. A different
      installed version is upgraded or downgraded to it.
    - Sets OpenCV_DIR to the OpenCV build tree and puts the OpenCV DLL folder on PATH.
      In GitHub Actions the values go to GITHUB_ENV and GITHUB_PATH; otherwise they
      are persisted for the chosen -Scope and set in the current session.

    Run it from an elevated (Administrator) PowerShell:
        powershell -ExecutionPolicy Bypass -File scripts\install-opencv-windows.ps1
    or from a normal one with -SelfElevate, which asks for Administrator rights
    through UAC. CMake does that when configured with
    -DAPPLICATION_SAMPLES_INSTALL_OPENCV=ON and OpenCV is missing.

.PARAMETER OpenCvVersion
    Chocolatey 'opencv' package version. This default is the single place where
    the Windows OpenCV version is pinned; CI uses this script as well.

.PARAMETER Scope
    Where OpenCV_DIR and PATH are persisted: User (default) or Machine.

.PARAMETER InstallRoot
    Folder the Chocolatey package unzips into. OpenCV lands in <InstallRoot>\opencv.
    cmake/CMakeLists.txt looks for OpenCV under the default C:\tools.

.PARAMETER SelfElevate
    When not elevated, start this script again as Administrator (UAC prompt),
    wait for it and show its output, instead of stopping with an error.

.PARAMETER TranscriptPath
    Internal: where the elevated copy started by -SelfElevate writes its output.
#>
[CmdletBinding()]
param(
    [string] $OpenCvVersion = '4.13.0',
    [ValidateSet('User', 'Machine')]
    [string] $Scope = 'User',
    [string] $InstallRoot = 'C:\tools',
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

function Invoke-Native {
    param([string] $Description, [scriptblock] $Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
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

# 1. Elevation: Chocolatey installs into Program Data and needs Administrator rights.
$principal = New-Object Security.Principal.WindowsPrincipal(
    [Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    if (-not $SelfElevate) {
        throw 'This script needs Administrator rights. Run PowerShell as Administrator and start it again, or add -SelfElevate.'
    }

    # The elevated copy runs in its own window, so it writes a transcript that
    # is shown here once it ends.
    $log = Join-Path ([IO.Path]::GetTempPath()) "install-opencv-windows-$PID.log"
    $childArgs = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
        '-OpenCvVersion', $OpenCvVersion, '-Scope', $Scope,
        '-InstallRoot', "`"$InstallRoot`"", '-TranscriptPath', "`"$log`"")
    Write-Host 'Asking for Administrator rights to install OpenCV...'
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

# 2. Chocolatey bootstrap (https://chocolatey.org/install).
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host 'Chocolatey not found; installing it.'
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    Invoke-Expression ((New-Object Net.WebClient).DownloadString(
        'https://community.chocolatey.org/install.ps1'))

    if (-not $env:ChocolateyInstall) {
        $env:ChocolateyInstall = [Environment]::GetEnvironmentVariable('ChocolateyInstall', 'Machine')
    }
    if (-not $env:ChocolateyInstall) {
        $env:ChocolateyInstall = Join-Path $env:ProgramData 'chocolatey'
    }
    $env:Path = "$env:ChocolateyInstall\bin;$env:Path"
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        throw "Chocolatey was installed but choco is not in $env:ChocolateyInstall\bin."
    }
}

# Collect the whole output first: piping into Select-Object -First stops choco
# early, and PowerShell 7 then leaves $LASTEXITCODE unset, which strict mode rejects.
$chocoOutput = @(& choco --version)
if ($LASTEXITCODE -ne 0) { throw "choco --version failed with exit code $LASTEXITCODE." }
$chocoVersion = "$($chocoOutput | Select-Object -First 1)".Trim()
if ([int]($chocoVersion -split '[.-]')[0] -lt 2) {
    throw "Chocolatey $chocoVersion is too old; version 2 or newer is required. Run 'choco upgrade chocolatey -y'."
}

# 3. OpenCV at exactly the pinned version.
$listed = & choco list opencv --exact --limit-output
if ($LASTEXITCODE -ne 0) { throw "choco list failed with exit code $LASTEXITCODE." }
$installedVersion = $null
foreach ($line in @($listed)) {
    $parts = "$line" -split '\|'
    if ($parts.Count -ge 2 -and $parts[0] -ieq 'opencv') { $installedVersion = $parts[1].Trim() }
}

$packageParams = "/InstallationPath:$InstallRoot"
if ($installedVersion -eq $OpenCvVersion) {
    Write-Host "OpenCV $OpenCvVersion is already installed."
}
elseif ($installedVersion) {
    Write-Warning "OpenCV $installedVersion is installed; changing it to the pinned version $OpenCvVersion."
    Invoke-Native 'choco upgrade opencv' {
        choco upgrade opencv --version $OpenCvVersion --allow-downgrade `
            --params $packageParams --no-progress -y
    }
}
else {
    Write-Host "Installing OpenCV $OpenCvVersion."
    Invoke-Native 'choco install opencv' {
        choco install opencv --version $OpenCvVersion --params $packageParams --no-progress -y
    }
}

# 4. OpenCV build tree and the newest Visual C++ DLL folder in it.
$openCvBuild = Join-Path $InstallRoot 'opencv\build'
$openCvBin = Get-ChildItem (Join-Path $openCvBuild 'x64\vc*\bin') -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $openCvBin) {
    throw "OpenCV binaries not found under $openCvBuild\x64. Was OpenCV installed into a different folder?"
}
$openCvBinDir = $openCvBin.FullName

# 5. Environment.
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

# 6. Summary.
Write-Host ''
Write-Host "OpenCV version : $OpenCvVersion"
Write-Host "OpenCV_DIR     : $openCvBuild"
Write-Host "OpenCV bin     : $openCvBinDir"

if ($TranscriptPath) {
    Stop-Transcript | Out-Null
}
