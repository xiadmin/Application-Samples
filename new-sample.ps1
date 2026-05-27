#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Interactive scaffold for a new Application-Samples entry.
.DESCRIPTION
    Walks through feature -> concrete-topic -> language selection,
    then generates the folder, README.md, CMakeLists.txt, and stub source.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot   = $PSScriptRoot
$samplesDir = Join-Path $repoRoot 'samples'

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

    $input   = ''
    $pattern = '^[a-z0-9]+(-[a-z0-9]+)*$'

    # Render the current input line in place.
    function Render-Input {
        param([string] $Text)
        $isValid = $Text -cmatch $pattern
        $hint    = if ($Text.Length -eq 0) { ' (lowercase letters, digits, hyphens)' }
                   elseif ($isValid)        { ' [valid]' }
                   else                     { ' [invalid -- lowercase letters/digits separated by single hyphens]' }
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
                if ($input -cmatch $pattern) {
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
                # Accept only printable ASCII; uppercase will show as invalid (red)
                if ($ch -ge 0x20 -and $ch -le 0x7E) {
                    $input += [char]$ch
                    Render-Input -Text $input
                }
            }
        }
    }
}

# ── step 1: feature ───────────────────────────────────────────────────────────

$features = @(Get-ChildItem -Path $samplesDir -Directory |
    Where-Object { $_.Name -match '^[a-z0-9]' } |
    Sort-Object Name |
    Select-Object -ExpandProperty Name)

$feature = Read-Selection -Items $features -Prompt 'Step 1 -- Select a feature folder:'
if ($null -eq $feature) {
    $feature = Read-KebabName -Prompt '  New feature name (kebab-case)'
}

# ── step 2: concrete topic ────────────────────────────────────────────────────

$featureDir = Join-Path $samplesDir $feature
$concretes  = @()
if (Test-Path $featureDir) {
    $concretes = @(Get-ChildItem -Path $featureDir -Directory |
        Where-Object { $_.Name -match '^[a-z0-9]' } |
        Sort-Object Name |
        Select-Object -ExpandProperty Name)
}

$topic = Read-Selection -Items $concretes -Prompt 'Step 2 -- Select a concrete topic:'
if ($null -eq $topic) {
    $topic = Read-KebabName -Prompt '  New topic name (kebab-case)'
}

# ── step 3: language ──────────────────────────────────────────────────────────

$langs = @('c', 'cpp', 'csharp')
$lang  = Read-Selection -Items $langs -Prompt 'Step 3 -- Select language:' -NoNew

# ── derive names ──────────────────────────────────────────────────────────────

$sampleDir   = Join-Path (Join-Path (Join-Path $samplesDir $feature) $topic) $lang
$binaryName  = "$feature-$topic-$lang"
$targetName  = $binaryName -replace '-', '_'
$cmakePath   = "samples/$feature/$topic/$lang"

# C / C++ specifics
$sourceFile  = if ($lang -eq 'c') { 'main.c' } else { 'main.cpp' }
$langStd     = if ($lang -eq 'c') { 'c_std_11' }  else { 'cxx_std_17' }
$projectLang = if ($lang -eq 'c') { 'C' }         else { 'CXX' }
$ximeaTarget = if ($lang -eq 'c') { 'XIMEA::xiAPI' } else { 'XIMEA::xiAPIplus' }

$langLabel   = switch ($lang) {
    'c'      { 'C' }
    'cpp'    { 'C++' }
    'csharp' { 'C#' }
}

# C# specifics — PascalCase project name derived from binary name segments
$csProjectName = ($binaryName -split '-' | ForEach-Object {
    $_.Substring(0,1).ToUpper() + $_.Substring(1)
}) -join ''

# ── confirm ───────────────────────────────────────────────────────────────────

Write-Host ''
Write-Host '-----------------------------------------' -ForegroundColor DarkGray
Write-Host " Feature  : $feature"
Write-Host " Topic    : $topic"
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
    Write-Host "Error: samples/$feature/$topic/$lang already exists. Nothing was created." -ForegroundColor Red
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

} else {

# ── CMakeLists.txt ────────────────────────────────────────────────────────────
# Single-quoted here-string = no interpolation; substitute placeholders after.

$cmakeTemplate = @'
cmake_minimum_required(VERSION 3.16)

project(%%BINARY_NAME%% VERSION 0.1.0 LANGUAGES %%PROJECT_LANG%%)

# Include shared CMake modules
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../../cmake")
include(SampleDefaults)
find_package(XIMEA REQUIRED)

add_executable(%%TARGET_NAME%% %%SOURCE_FILE%%)
target_compile_features(%%TARGET_NAME%% PRIVATE %%LANG_STD%%)
target_link_libraries(%%TARGET_NAME%% PRIVATE %%XIMEA_TARGET%%)

# Apply output directory flattening to this target
sample_flat_output_directories(TARGET %%TARGET_NAME%%)
'@

$cmakeContent = $cmakeTemplate `
    -replace '%%BINARY_NAME%%',  $binaryName `
    -replace '%%PROJECT_LANG%%', $projectLang `
    -replace '%%TARGET_NAME%%',  $targetName `
    -replace '%%SOURCE_FILE%%',  $sourceFile `
    -replace '%%LANG_STD%%',     $langStd `
    -replace '%%XIMEA_TARGET%%', $ximeaTarget

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
} else {
    Write-Host "  1. Fill in the TODO sections in $sourceFile"
    Write-Host '  2. Update README.md (description, expected output, limitations)'
    Write-Host "  3. Build: cd $cmakePath && cmake -B .cmake-tmp && cmake --build .cmake-tmp"
}
Write-Host ''
