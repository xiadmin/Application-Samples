#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Interactive scaffold for a new Application-Samples entry.
.DESCRIPTION
    Walks through API -> (Cross-Platform | Hardware-specific, XiAPI only) ->
    sample-name -> language selection, then generates the folder, README.md,
    CMakeLists.txt / csproj, and stub source.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot   = $PSScriptRoot
$samplesDir = Join-Path $repoRoot 'Samples'

# ── helpers ───────────────────────────────────────────────────────────────────

# Renders the menu, highlighting the current selection.
# allItems = existing items + a sentinel '+ Create new' entry at the end.
function Render-Menu {
    param(
        [string]   $Prompt,
        [string[]] $AllItems,
        [int]      $Selected,
        [bool]     $HasNew = $false
    )
    # Move cursor back to the top of the menu each redraw.
    # On the first draw $script:menuLines is 0 so nothing moves.
    if ($script:menuLines -gt 0) {
        $host.UI.RawUI.CursorPosition = New-Object System.Management.Automation.Host.Coordinates `
            0, ($host.UI.RawUI.CursorPosition.Y - $script:menuLines)
    }
    Write-Host ''
    Write-Host $Prompt -ForegroundColor Cyan
    for ($i = 0; $i -lt $AllItems.Count; $i++) {
        if ($i -eq $Selected) {
            Write-Host ("  > $($AllItems[$i])") -ForegroundColor Green
        } elseif ($HasNew -and $i -eq $AllItems.Count - 1) {
            Write-Host ("    $($AllItems[$i])") -ForegroundColor Yellow
        } else {
            Write-Host ("    $($AllItems[$i])")
        }
    }
    Write-Host ''
    # lines drawn: blank + header + items + blank
    $script:menuLines = 2 + $AllItems.Count + 1
}

# Arrow-key menu. Returns the selected existing item, or $null for 'Create new'.
# Pass -NoNew to hide the 'Create new' option entirely (fixed lists).
function Read-Selection {
    param(
        [string[]] $Items,
        [string]   $Prompt,
        [string]   $NewLabel = 'Create new',
        [switch]   $NoNew
    )

    if ($NoNew) { [string[]] $allItems = @($Items) }
    else         { [string[]] $allItems = @($Items) + @("+ $NewLabel") }
    $selected = 0
    $script:menuLines = 0
    $hasNew = -not $NoNew

    [Console]::CursorVisible = $false
    try {
        Render-Menu -Prompt $Prompt -AllItems $allItems -Selected $selected -HasNew $hasNew
        while ($true) {
            $key = $host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
            switch ($key.VirtualKeyCode) {
                38 {  # Up
                    if ($selected -gt 0) { $selected-- }
                    Render-Menu -Prompt $Prompt -AllItems $allItems -Selected $selected -HasNew $hasNew
                }
                40 {  # Down
                    if ($selected -lt $allItems.Count - 1) { $selected++ }
                    Render-Menu -Prompt $Prompt -AllItems $allItems -Selected $selected -HasNew $hasNew
                }
                13 {  # Enter
                    Write-Host ''
                    if (-not $NoNew -and $selected -eq $allItems.Count - 1) { return $null }
                    return $Items[$selected]
                }
            }
        }
    } finally {
        [Console]::CursorVisible = $true
    }
}

function Read-KebabName {
    param([string] $Prompt)

    # Naming convention for API/sample folders is officially TBD (see design doc), and
    # existing folders already mix PascalCase, dots, '#', and spaces (e.g. "XiAPI.NET-C#",
    # "GPIO samples-Jetson"). We only reject characters that are illegal in file/folder
    # names on Windows or Linux, rather than forcing a specific case convention.
    $input   = ''
    $pattern = '^[^\\/:\*\?"<>\|]+$'

    # Render the current input line in place.
    function Render-Input {
        param([string] $Text)
        $isValid = $Text.Length -gt 0 -and $Text -notmatch '[\\/:\*\?"<>\|]' -and $Text.Trim() -eq $Text
        $hint    = if ($Text.Length -eq 0) { ' (any valid folder name)' }
                   elseif ($isValid)        { ' [valid]' }
                   else                     { ' [invalid -- no \ / : * ? " < > | and no leading/trailing spaces]' }
        $color   = if ($isValid) { 'Green' } else { 'Red' }

        # Overwrite the line
        $width = $host.UI.RawUI.WindowSize.Width
        Write-Host ("`r" + (' ' * ($width - 1)) + "`r") -NoNewline
        Write-Host "$Prompt`: " -NoNewline -ForegroundColor Cyan
        Write-Host $Text -NoNewline -ForegroundColor $color
        Write-Host $hint -NoNewline -ForegroundColor DarkGray
    }

    Write-Host ''
    [Console]::CursorVisible = $true
    Render-Input -Text $input

    while ($true) {
        $key = $host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

        switch ($key.VirtualKeyCode) {
            13 {  # Enter
                if ($input -match $pattern -and $input.Trim() -eq $input) {
                    Write-Host ''
                    return $input
                }
                # flash hint — already shown, just beep
                [Console]::Beep(400, 150)
            }
            8 {   # Backspace
                if ($input.Length -gt 0) { $input = $input.Substring(0, $input.Length - 1) }
                Render-Input -Text $input
            }
            27 {  # Escape — clear field
                $input = ''
                Render-Input -Text $input
            }
            default {
                $ch = $key.Character
                # Accept printable ASCII (and beyond); illegal path characters are filtered on Enter
                if ($ch -ge 0x20) {
                    $input += [char]$ch
                    Render-Input -Text $input
                }
            }
        }
    }
}

# ── step 1: API ───────────────────────────────────────────────────────────────
# Samples/<API>/... -- e.g. XiAPI, XiAPI.NET-C#, XiApiPython. XiAPI additionally
# splits into Cross-Platform / Hardware-specific (see repository layout in README.md).

$apis = @(Get-ChildItem -Path $samplesDir -Directory |
    Sort-Object Name |
    Select-Object -ExpandProperty Name)

$api = Read-Selection -Items $apis -Prompt 'Step 1 -- Select an API folder:'
if ($null -eq $api) {
    $api = Read-KebabName -Prompt '  New API folder name'
}

# ── step 2: Cross-Platform / Hardware-specific (XiAPI only), then sample name ──

$group = $null
if ($api -eq 'XiAPI') {
    $group = Read-Selection -Items @('Cross-Platform', 'Hardware-specific') -Prompt 'Step 2 -- Select sample group:' -NoNew
    $parentDir = Join-Path (Join-Path $samplesDir $api) $group
} else {
    $parentDir = Join-Path $samplesDir $api
}

$samples = @()
if (Test-Path $parentDir) {
    $samples = @(Get-ChildItem -Path $parentDir -Directory |
        Sort-Object Name |
        Select-Object -ExpandProperty Name)
}

$stepNum = if ($api -eq 'XiAPI') { 'Step 3' } else { 'Step 2' }
$topic = Read-Selection -Items $samples -Prompt "$stepNum -- Select a sample:"
if ($null -eq $topic) {
    $topic = Read-KebabName -Prompt '  New sample name'
}

# ── step 3/4: language ────────────────────────────────────────────────────────
# XiAPI Cross-Platform samples pick c/cpp and get a language subfolder (the same
# sample is offered in both languages). XiAPI Hardware-specific samples are tied
# to one specific implementation, so the language only picks the source stub --
# no extra subfolder. Other APIs imply their language from the API folder itself.

$langStepNum = if ($api -eq 'XiAPI') { 'Step 4' } else { 'Step 3' }
if ($api -eq 'XiAPI') {
    $lang = Read-Selection -Items @('c', 'cpp') -Prompt "$langStepNum -- Select language:" -NoNew
} elseif ($api -eq 'XiAPI.NET-C#') {
    $lang = 'csharp'
} elseif ($api -eq 'XiApiPython') {
    $lang = 'python'
} else {
    Write-Host ''
    Write-Host "  '$api' is a new API folder -- pick the closest matching language template:" -ForegroundColor DarkGray
    $lang = Read-Selection -Items @('c', 'cpp', 'csharp', 'python') -Prompt "$langStepNum -- Select language:" -NoNew
}

# ── derive names ──────────────────────────────────────────────────────────────

if ($api -eq 'XiAPI' -and $group -eq 'Cross-Platform') {
    # Cross-Platform: Samples/XiAPI/Cross-Platform/<sample-name>/<lang>/
    $sampleDir  = Join-Path (Join-Path (Join-Path (Join-Path $samplesDir $api) $group) $topic) $lang
    $folderName = "$api-$group-$topic-$lang"
    $cmakePath  = "Samples/$api/$group/$topic/$lang"
} elseif ($api -eq 'XiAPI') {
    # Hardware-specific: Samples/XiAPI/Hardware-specific/<sample-name>/ (no lang subfolder)
    $sampleDir  = Join-Path (Join-Path (Join-Path $samplesDir $api) $group) $topic
    $folderName = "$api-$group-$topic"
    $cmakePath  = "Samples/$api/$group/$topic"
} else {
    # Other APIs: Samples/<API>/<sample-name>/ (language implied by the API folder)
    $sampleDir  = Join-Path (Join-Path $samplesDir $api) $topic
    $folderName = "$api-$topic"
    $cmakePath  = "Samples/$api/$topic"
}

# Relative path from the sample folder up to the repo root's shared cmake/ folder,
# computed from path depth so it stays correct regardless of API/group nesting.
$cmakeDepth       = ($cmakePath -split '/').Count
$cmakeIncludePath = ((@('..') * $cmakeDepth) -join '/') + '/cmake'

# Binary/assembly name: for compiled C/C++ samples this must match the OUTPUT_NAME
# that cmake/SampleDefaults.cmake derives from the full folder path (== $folderName).
# For C#/Python it is just a display name, kept short and independent of the API
# folder (which may contain characters like '.' or '#' that don't belong in an
# assembly name), matching the convention used by existing samples.
$binaryName = if ($lang -eq 'c' -or $lang -eq 'cpp') { $folderName } else { "$topic-$lang" }
$targetName = $binaryName -replace '[^a-zA-Z0-9]', '_'

# C / C++ specifics
$sourceFile  = if ($lang -eq 'c') { 'main.c' } else { 'main.cpp' }
$langStd     = if ($lang -eq 'c') { 'c_std_11' }  else { 'cxx_std_17' }
$projectLang = if ($lang -eq 'c') { 'C' }         else { 'CXX' }
$ximeaTarget = if ($lang -eq 'c') { 'XIMEA::xiAPI' } else { 'XIMEA::xiAPIplus' }

$langLabel   = switch ($lang) {
    'c'      { 'C' }
    'cpp'    { 'C++' }
    'csharp' { 'C#' }
    'python' { 'Python' }
}

# C# specifics — PascalCase project name derived from binary name segments
$csProjectName = ($binaryName -split '[^a-zA-Z0-9]+' | Where-Object { $_.Length -gt 0 } | ForEach-Object {
    $_.Substring(0,1).ToUpper() + $_.Substring(1)
}) -join ''

# ── confirm ───────────────────────────────────────────────────────────────────

Write-Host ''
Write-Host '-----------------------------------------' -ForegroundColor DarkGray
Write-Host " API      : $api"
if ($group) { Write-Host " Group    : $group" }
Write-Host " Sample   : $topic"
Write-Host " Language : $lang"
Write-Host " Path     : $cmakePath"
Write-Host " Binary   : $binaryName"
Write-Host '-----------------------------------------' -ForegroundColor DarkGray
Write-Host ''
$confirm = Read-Selection -Items @('Yes', 'No') -Prompt 'Create this sample?' -NoNew
if ($confirm -ne 'Yes') {
    Write-Host 'Aborted.' -ForegroundColor Yellow
    exit 0
}

# ── guard: already exists ─────────────────────────────────────────────────────

if (Test-Path $sampleDir) {
    Write-Host ''
    Write-Host "Error: $cmakePath already exists. Nothing was created." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Path $sampleDir -Force | Out-Null

if ($lang -eq 'csharp') {

# ── C# scaffold ───────────────────────────────────────────────────────────────

$emDash    = [char]0x2014
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

$csprojContent = @"
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <!-- Target latest LTS .NET; update when a new LTS ships. -->
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <AssemblyName>$binaryName</AssemblyName>
    <RootNamespace>Ximea.Samples</RootNamespace>
    <!-- Force x64 $emDash xiApi.NETX64.dll is 64-bit only. -->
    <PlatformTarget>x64</PlatformTarget>
    <AllowUnsafeBlocks>false</AllowUnsafeBlocks>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <!--
      Resolve SDK location from either:
        - Command-line override:  dotnet build -p:XimeaSpPath=D:\XIMEA
        - Environment variable:  XIMEA_SP_PATH (set by the XIMEA installer, default C:\XIMEA)
      XimeaSpPath takes precedence when both are present.
    -->
    <XimeaSpPath Condition="'`$(XimeaSpPath)' == ''">`$(XIMEA_SP_PATH)</XimeaSpPath>
  </PropertyGroup>

  <ItemGroup>
    <Reference Include="xiApi.NETX64">
      <!-- XIMEA ships up to net7.0; the net7.0 assembly is forward-compatible with net8.0. -->
      <HintPath>`$(XimeaSpPath)\API\xiAPI.NET.NET.7.0\xiApi.NETX64.dll</HintPath>
    </Reference>
  </ItemGroup>

  <!-- Fail fast with an actionable message when no SDK path could be resolved. -->
  <Target Name="CheckXimeaSpPath" BeforeTargets="Build">
    <Error Condition="'`$(XimeaSpPath)' == ''"
           Text="XIMEA SDK path is not set. Run the XIMEA installer (which sets XIMEA_SP_PATH) or pass -p:XimeaSpPath=&lt;path&gt; to dotnet build." />
  </Target>

</Project>
"@

[System.IO.File]::WriteAllText((Join-Path $sampleDir "$csProjectName.csproj"), $csprojContent, $utf8NoBom)

$programContent = @"
#nullable enable

using System;
using xiApi.NET;

// TODO: describe what this sample does.

var cam = new xiCam();
bool isDeviceOpen = false;
try
{
    cam.GetNumberDevices(out int numDevices);

    if (numDevices == 0)
    {
        Console.Error.WriteLine("Error: no XIMEA cameras detected.");
        return 1;
    }

    Console.WriteLine(`$"Found {numDevices} camera(s), opening index 0.");
    cam.OpenDevice(0);
    isDeviceOpen = true;

    // TODO: implement
    Console.WriteLine("${binaryName}: not yet implemented");
    return 0;
}
catch (xiExc ex)
{
    Console.Error.WriteLine(`$"Error: {ex.Message}");
    return 1;
}
finally
{
    if (isDeviceOpen)
        cam.CloseDevice();
}
"@

[System.IO.File]::WriteAllText((Join-Path $sampleDir 'Program.cs'), $programContent, $utf8NoBom)

$csReadmeContent = @"
# $topic $emDash C# sample

TODO: one-line description of what this sample demonstrates.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.32 or newer |
| .NET SDK | 8.0 or newer |

---

## Build

### Using build.ps1 (builds all samples)

``````powershell
cd <repo-root>
.\build.ps1
``````

Binary and supporting files land in:

``````
build\$binaryName\
``````

### Directly with dotnet

``````powershell
cd $cmakePath
dotnet build $csProjectName.csproj -c Release --output .dotnet-tmp
``````

Binary lands in ``.dotnet-tmp\\`` inside the sample folder.

---

## Run

### After build.ps1

``````powershell
.\build\$binaryName\$binaryName.exe
``````

### After a direct dotnet build

``````powershell
.\\$cmakePath\\.dotnet-tmp\\$binaryName.exe
``````

Or use ``dotnet run`` (no separate build step needed):

``````powershell
cd $cmakePath
dotnet run --project $csProjectName.csproj
``````

---

## Expected output

``````
TODO: paste expected console output here.
``````

---

## Known limitations / caveats

- Windows-only: the XIMEA .NET wrapper is not available for Linux or macOS.
- The project targets net8.0 but links against the net7.0 ``xiApi.NETX64.dll`` (the latest
  version shipped with the SDK). Forward compatibility is supported by the .NET runtime.

---

## Links

- [xiAPI.NET documentation](https://www.ximea.com/support/wiki/apis/XiAPINET_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
"@

[System.IO.File]::WriteAllText((Join-Path $sampleDir 'README.md'), $csReadmeContent, $utf8NoBom)

} elseif ($lang -eq 'python') {

# ── Python scaffold ────────────────────────────────────────────────────────────

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
$emDash    = [char]0x2014

$mainContent = @"
"""${binaryName} -- XIMEA xiAPI sample (Python)

TODO: describe what this sample does.
"""

import ximea.xiapi as xiapi


def main() -> int:
    cam = xiapi.Camera()
    cam.open_device()

    # TODO: implement
    print("${binaryName}: not yet implemented", flush=True)

    cam.close_device()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"@

[System.IO.File]::WriteAllText((Join-Path $sampleDir 'main.py'), $mainContent, $utf8NoBom)

$pyReadmeContent = @"
# $topic $emDash Python sample

TODO: one-line description of what this sample demonstrates.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | 4.33 or newer |
| Python | 3.9 or newer |

No separate pip install is needed -- the ``ximea`` package must be placed into ``site-packages/ximea``.

---

## Run

### Directly

``````bash
# Linux / macOS
python $cmakePath/main.py

# Windows PowerShell
python $cmakePath\main.py
``````

### After build.ps1 (Windows)

``````powershell
.\build\$binaryName\run.ps1
``````

---

## Expected output

``````
TODO: paste expected console output here.
``````

---

## Known limitations / caveats

-

---

## Links

- [XIMEA Python API documentation](https://www.ximea.com/support/wiki/apis/XiAPI_Python_Manual)
- [XIMEA Software Packages](https://www.ximea.com/software-downloads)
"@

[System.IO.File]::WriteAllText((Join-Path $sampleDir 'README.md'), $pyReadmeContent, $utf8NoBom)

} else {

# ── CMakeLists.txt ────────────────────────────────────────────────────────────
# Single-quoted here-string = no interpolation; substitute placeholders after.

$cmakeTemplate = @'
cmake_minimum_required(VERSION 3.16)

project(%%BINARY_NAME%% VERSION 0.1.0 LANGUAGES %%PROJECT_LANG%%)

# Include shared CMake modules
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/%%CMAKE_INCLUDE_PATH%%")
include(SampleDefaults)
find_package(XIMEA REQUIRED)

add_executable(%%TARGET_NAME%% %%SOURCE_FILE%%)
target_compile_features(%%TARGET_NAME%% PRIVATE %%LANG_STD%%)
target_link_libraries(%%TARGET_NAME%% PRIVATE %%XIMEA_TARGET%%)

# Apply output directory flattening to this target
sample_flat_output_directories(TARGET %%TARGET_NAME%%)
'@

$cmakeContent = $cmakeTemplate `
    -replace '%%BINARY_NAME%%',       $binaryName `
    -replace '%%PROJECT_LANG%%',      $projectLang `
    -replace '%%TARGET_NAME%%',       $targetName `
    -replace '%%SOURCE_FILE%%',       $sourceFile `
    -replace '%%LANG_STD%%',          $langStd `
    -replace '%%XIMEA_TARGET%%',      $ximeaTarget `
    -replace '%%CMAKE_INCLUDE_PATH%%', $cmakeIncludePath

Set-Content -Path (Join-Path $sampleDir 'CMakeLists.txt') -Value $cmakeContent -Encoding UTF8

# ── source stub ───────────────────────────────────────────────────────────────

if ($lang -eq 'c') {
    $sourceTemplate = @'
/* %%TARGET_NAME%% -- XIMEA xiAPI sample (C)
 *
 * TODO: describe what this sample does.
 *
 * Build: see CMakeLists.txt or build.ps1 at the repo root.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xiApi.h>

int main(void)
{
    /* TODO: implement */
    printf("%%BINARY_NAME%%: not yet implemented\n");
    return EXIT_SUCCESS;
}
'@
} else {
    $sourceTemplate = @'
// %%TARGET_NAME%% -- XIMEA xiAPIplus sample (C++)
//
// TODO: describe what this sample does.
//
// Build: see CMakeLists.txt or build.ps1 at the repo root.

#include <cstdlib>
#include <iostream>
#include <xiApiPlus.h>

int main()
{
    // TODO: implement
    std::cout << "%%BINARY_NAME%%: not yet implemented\n";
    return EXIT_SUCCESS;
}
'@
}

$sourceContent = $sourceTemplate `
    -replace '%%TARGET_NAME%%', $targetName `
    -replace '%%BINARY_NAME%%', $binaryName

Set-Content -Path (Join-Path $sampleDir $sourceFile) -Value $sourceContent -Encoding UTF8

# ── README.md ─────────────────────────────────────────────────────────────────

$readmeTemplate = @'
# %%TOPIC%% — %%LANG_LABEL%% sample

TODO: one-line description of what this sample demonstrates.

---

## Prerequisites

| Item | Requirement |
|------|-------------|
| OS | Windows 10/11 or Linux (Ubuntu 20.04+) |
| Hardware | Any XIMEA USB3 / PCIe camera |
| XIMEA SDK | tested with 4.33 |
| CMake | 3.16 or newer |
| Compiler | MSVC 2022+, GCC 9+, or Clang 10+ |

---

## Build

Build from the sample folder using CMake directly, or use the PowerShell
helper at the repo root which builds all samples in one shot.

### CMake directly — Linux

```bash
cd %%CMAKE_PATH%%
cmake -B .cmake-tmp
cmake --build .cmake-tmp
```

Binary lands in `.cmake-tmp/build/`.

### CMake directly — Windows (PowerShell)

```powershell
cd %%CMAKE_PATH%%
cmake -B .cmake-tmp -A x64
cmake --build .cmake-tmp --config Release
```

Binary lands in `.cmake-tmp\build\Release\`.

## Run

After a direct CMake build:

```bash
# Linux
.cmake-tmp/build/%%BINARY_NAME%%

# Windows PowerShell
.\.cmake-tmp\build\Release\%%BINARY_NAME%%.exe
```

---

## Expected output

```
TODO: paste expected console output here.
```

---

## Known limitations / caveats

-

---

## Links

- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)
'@

$readmeContent = $readmeTemplate `
    -replace '%%TOPIC%%',       $topic `
    -replace '%%LANG_LABEL%%',  $langLabel `
    -replace '%%CMAKE_PATH%%',  $cmakePath `
    -replace '%%BINARY_NAME%%', $binaryName

if ($lang -eq 'cpp') {
    $readmeContent = $readmeContent -replace `
        '\- \[xiAPI documentation\]\(https://www\.ximea\.com/support/wiki/apis/xiapi_manual\)', `
        "- [xiAPIplus documentation](https://www.ximea.com/support/wiki/apis/XIMEA_xiAPIplus_Page)`n- [xiAPI documentation](https://www.ximea.com/support/wiki/apis/xiapi_manual)"
}

Set-Content -Path (Join-Path $sampleDir 'README.md') -Value $readmeContent -Encoding UTF8

} # end if csharp / else

# ── done ──────────────────────────────────────────────────────────────────────

Write-Host ''
Write-Host 'Sample scaffold created:' -ForegroundColor Green
Get-ChildItem -Path $sampleDir | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Cyan
if ($lang -eq 'csharp') {
    Write-Host '  1. Fill in the TODO sections in Program.cs'
    Write-Host '  2. Update README.md (description, expected output, limitations)'
    Write-Host "  3. Build: cd $cmakePath && dotnet build $csProjectName.csproj"
} elseif ($lang -eq 'python') {
    Write-Host '  1. Fill in the TODO sections in main.py'
    Write-Host '  2. Update README.md (description, expected output, limitations)'
    Write-Host "  3. Run: python $cmakePath/main.py"
    Write-Host '  (No pip install needed -- ximea is installed by the XIMEA SDK into site-packages)'
} else {
    Write-Host "  1. Fill in the TODO sections in $sourceFile"
    Write-Host '  2. Update README.md (description, expected output, limitations)'
    Write-Host "  3. Build: cd $cmakePath && cmake -B .cmake-tmp && cmake --build .cmake-tmp"
}
Write-Host ''
