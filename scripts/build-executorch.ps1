[CmdletBinding()]
param(
    [ValidateSet('Host', 'ARM64')][string]$Target = 'Host',
    [string]$QnnSdk = 'C:\Qualcomm\AIStack\QAIRT\2.50.40.260831',
    [int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $projectRoot '.deps\executorch'
$buildRoot = Join-Path $projectRoot ('build\executorch-' + $Target.ToLower())
$pythonExe = & python -c 'import sys; print(sys.executable)'
$pythonVersion = & $pythonExe -c 'import sys; print("%d.%d" % sys.version_info[:2])'
if ($pythonVersion -ne '3.14') { throw 'Use the installed Python 3.14 for this project.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot = & $vswhere -latest -products '*' -property installationPath
$devShell = Join-Path $vsRoot 'Common7\Tools\Launch-VsDevShell.ps1'
$targetArch = if ($Target -eq 'Host') { 'amd64' } else { 'arm64' }
& $devShell -Arch $targetArch -HostArch amd64 -SkipAutomaticLocation
$env:QNN_SDK_ROOT = $QnnSdk
$env:PATH = (Split-Path -Parent $pythonExe) + ';' + (Join-Path (Split-Path -Parent $pythonExe) 'Scripts') + ';' + $env:PATH
$cmakePlatform = if ($Target -eq 'Host') { 'x64' } else { 'ARM64' }
$processor = if ($Target -eq 'Host') { 'AMD64' } else { 'ARM64' }
$configArgs = @(
    '-S', $sourceRoot, '-B', $buildRoot,
    '-G', 'Visual Studio 17 2022', '-A', $cmakePlatform,
    '-DCMAKE_SYSTEM_NAME=Windows', "-DCMAKE_SYSTEM_PROCESSOR=$processor",
    '-DCMAKE_BUILD_TYPE=Release', "-DPYTHON_EXECUTABLE=$pythonExe", "-DPython_EXECUTABLE=$pythonExe",
    "-DQNN_SDK_ROOT=$QnnSdk", "-DCMAKE_INSTALL_PREFIX=$buildRoot/install",
    '-DEXECUTORCH_BUILD_QNN=ON', '-DEXECUTORCH_BUILD_XNNPACK=OFF',
    '-DEXECUTORCH_BUILD_PTHREADPOOL=OFF', '-DEXECUTORCH_BUILD_CPUINFO=OFF',
    '-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF', '-DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON',
    '-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON', '-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON',
    '-DEXECUTORCH_ENABLE_LOGGING=ON', '-DBUILD_TESTING=OFF'
)
& cmake @configArgs
if ($LASTEXITCODE) { throw 'ExecuTorch configure failed.' }
$buildTarget = if ($Target -eq 'Host') { 'PyQnnManagerAdaptor' } else { 'install' }
& cmake --build $buildRoot --config Release --parallel $Jobs --target $buildTarget
if ($LASTEXITCODE) { throw 'ExecuTorch build failed.' }
if ($Target -eq 'Host') {
    $modules = Get-ChildItem -LiteralPath (Join-Path $buildRoot 'backends\qualcomm\Release') -Filter 'PyQnnManagerAdaptor*.pyd'
    if (-not $modules) { throw 'QNN Python extension was not built.' }
    $modules | Copy-Item -Destination (Join-Path $sourceRoot 'backends\qualcomm\python')
}
