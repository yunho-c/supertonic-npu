[CmdletBinding()]
param(
    [string]$QnnSdk = 'C:\Qualcomm\AIStack\QAIRT\2.50.40.260831',
    [int]$Jobs = 8
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& python -c 'import sys; assert sys.version_info[:2] == (3, 14); assert sys.prefix == sys.base_prefix, "Use global Python, without a virtualenv"; print(sys.executable)'
if ($LASTEXITCODE) { throw 'Python 3.14 without a virtualenv is required.' }
$sourceRoot = Join-Path $projectRoot '.deps/executorch'
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    & git clone --depth 1 --branch v1.5.0 https://github.com/pytorch/executorch.git $sourceRoot
    if ($LASTEXITCODE) { throw 'ExecuTorch clone failed.' }
}
$revision = & git -C $sourceRoot rev-parse HEAD
if ($revision -ne 'f7140a46ff38e919c557d45b102d3ff26097c8c9') { throw 'Unexpected ExecuTorch revision.' }
& git -C $sourceRoot submodule update --init --depth 1 -- third-party/flatbuffers third-party/flatcc third-party/gflags third-party/json third-party/pybind11 backends/xnnpack/third-party/FXdiv
if ($LASTEXITCODE) { throw 'Build dependency checkout failed.' }
& git submodule update --init external/supertonic
if ($LASTEXITCODE) { throw 'Supertonic checkout failed.' }
& python -m pip install --no-cache-dir -r requirements.txt
if ($LASTEXITCODE) { throw 'Python package installation failed.' }
& python scripts/prepare_executorch.py
if ($LASTEXITCODE) { throw 'ExecuTorch preparation failed.' }
& python scripts/download_assets.py
if ($LASTEXITCODE) { throw 'Asset verification failed.' }
& "$PSScriptRoot/build-executorch.ps1" -Target Host -QnnSdk $QnnSdk -Jobs $Jobs
& "$PSScriptRoot/build-executorch.ps1" -Target ARM64 -QnnSdk $QnnSdk -Jobs $Jobs
& "$PSScriptRoot/build-runner.ps1" -Jobs $Jobs
