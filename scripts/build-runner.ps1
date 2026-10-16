[CmdletBinding()]
param([int]$Jobs = 8)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -property installationPath
& (Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1') -Arch arm64 -HostArch amd64 -SkipAutomaticLocation
& cmake -S (Join-Path $projectRoot 'native') -B (Join-Path $projectRoot 'build/runner') `
    -G 'Visual Studio 17 2022' -A ARM64 "-DCMAKE_PREFIX_PATH=$projectRoot/build/executorch-arm64/install"
if ($LASTEXITCODE) { throw 'Runner configure failed.' }
& cmake --build (Join-Path $projectRoot 'build/runner') --config Release --parallel $Jobs
if ($LASTEXITCODE) { throw 'Runner build failed.' }
