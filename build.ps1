$ErrorActionPreference = "Stop"

$rootDir = $PSScriptRoot
if ([string]::IsNullOrEmpty($rootDir)) {
    $rootDir = Get-Location
}

$samplesRoot     = Join-Path $rootDir "samples"
$finalBuildRoot  = Join-Path $rootDir "build"
$cmakeTmpRoot    = Join-Path $rootDir ".cmake-tmp"
$dotnetTmpRoot   = Join-Path $rootDir ".dotnet-tmp"

if (Test-Path $cmakeTmpRoot)  { Remove-Item -Recurse -Force $cmakeTmpRoot }
if (Test-Path $dotnetTmpRoot) { Remove-Item -Recurse -Force $dotnetTmpRoot }
if (Test-Path $finalBuildRoot){ Remove-Item -Recurse -Force $finalBuildRoot }

$totalFound = 0
$totalOk    = 0
$failed     = [System.Collections.Generic.List[string]]::new()

# ── C / C++ samples (CMakeLists.txt in a leaf folder named 'c' or 'cpp') ──────

Write-Host "Finding C and C++ samples in $samplesRoot..."

$cmakeFiles = Get-ChildItem -Path $samplesRoot -Filter "CMakeLists.txt" -Recurse

foreach ($cmakeFile in $cmakeFiles) {
    $sampleDir = $cmakeFile.Directory.FullName

    # Only process C and C++ sample folders (language is the leaf directory)
    if ($sampleDir -match '[\/\\]c(pp)?$') {
        $totalFound++

        # Extract relative path from samples root
        $relativePath = $sampleDir.Substring($samplesRoot.Length).Trim('\', '/')
        # Convert slashes to hyphens
        $folderName = $relativePath -replace '[\/\\]', '-'

        Write-Host "=========================================="
        Write-Host "Building: $folderName"
        Write-Host "=========================================="

        $tmpBuildDir    = Join-Path $cmakeTmpRoot $folderName
        $targetBuildDir = Join-Path $finalBuildRoot $folderName

        # Configure
        & cmake -S $sampleDir -B $tmpBuildDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CMake configure failed for $folderName"
            $failed.Add($folderName)
            continue
        }

        # Build
        & cmake --build $tmpBuildDir --config Release
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "CMake build failed for $folderName"
            $failed.Add($folderName)
            continue
        }

        # Create destination folder
        New-Item -ItemType Directory -Force -Path $targetBuildDir | Out-Null

        # Copy binaries
        # First check multi-config (e.g. Visual Studio puts in build/Release)
        $builtBinDir     = Join-Path $tmpBuildDir "build"
        $builtReleaseDir = Join-Path $builtBinDir "Release"

        if (Test-Path $builtReleaseDir) {
            Get-ChildItem -Path $builtReleaseDir -File | ForEach-Object {
                Copy-Item -Path $_.FullName -Destination $targetBuildDir -Force
            }
        } elseif (Test-Path $builtBinDir) {
            # Single-config (e.g. Makefiles puts in build/)
            Get-ChildItem -Path $builtBinDir -File | ForEach-Object {
                Copy-Item -Path $_.FullName -Destination $targetBuildDir -Force
            }
        } else {
            # Fallback if RUNTIME_OUTPUT_DIRECTORY didn't catch it
            Get-ChildItem -Path $tmpBuildDir -Recurse -File | Where-Object {
                ($_.Extension -eq ".exe") -or ($_.Extension -eq "" -and $_.FullName -notmatch "CMakeFiles|Makefile")
            } | ForEach-Object {
                Copy-Item -Path $_.FullName -Destination $targetBuildDir -Force
            }
        }

        $totalOk++
    }
}

# Clean up CMake temporary files
Write-Host "=========================================="
Write-Host "Cleaning up CMake temporary files..."
if (Test-Path $cmakeTmpRoot) { Remove-Item -Recurse -Force $cmakeTmpRoot }

# ── C# samples (.csproj in a leaf folder named 'csharp') ──────────────────────

Write-Host ""
Write-Host "Finding C# samples in $samplesRoot..."

$csprojFiles = Get-ChildItem -Path $samplesRoot -Filter "*.csproj" -Recurse

foreach ($csprojFile in $csprojFiles) {
    $sampleDir = $csprojFile.Directory.FullName

    if ($sampleDir -match '[\/\\]csharp$') {
        $totalFound++

        $relativePath = $sampleDir.Substring($samplesRoot.Length).Trim('\', '/')
        $folderName   = $relativePath -replace '[\/\\]', '-'

        Write-Host "=========================================="
        Write-Host "Building: $folderName"
        Write-Host "=========================================="

        $tmpBuildDir    = Join-Path $dotnetTmpRoot $folderName
        $targetBuildDir = Join-Path $finalBuildRoot $folderName

        # Build and output into the temp directory so nothing lands in the source tree
        & dotnet build $csprojFile.FullName -c Release --output $tmpBuildDir -p:BaseIntermediateOutputPath="$tmpBuildDir\obj\\"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "dotnet build failed for $folderName"
            $failed.Add($folderName)
            continue
        }

        New-Item -ItemType Directory -Force -Path $targetBuildDir | Out-Null

        # Copy everything dotnet produced (exe, dll, runtime config, pdb)
        Get-ChildItem -Path $tmpBuildDir -File | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination $targetBuildDir -Force
        }

        $totalOk++
    }
}

# Clean up dotnet temporary files
Write-Host "=========================================="
Write-Host "Cleaning up dotnet temporary files..."
if (Test-Path $dotnetTmpRoot) { Remove-Item -Recurse -Force $dotnetTmpRoot }

# ── summary ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Build summary"                           -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Samples found  : $totalFound"
Write-Host "  Succeeded      : $totalOk"    -ForegroundColor $(if ($totalOk -eq $totalFound) { 'Green' } else { 'Yellow' })
Write-Host "  Failed         : $($failed.Count)" -ForegroundColor $(if ($failed.Count -eq 0) { 'Green' } else { 'Red' })
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed samples:" -ForegroundColor Red
    foreach ($name in $failed) {
        Write-Host "    - $name" -ForegroundColor Red
    }
}
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
