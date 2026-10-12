# Downloads a GPL FFmpeg build (with vid.stab and zimg, needed for stabilization
# and HDR conversion) into apps/desktop/src-tauri/ffmpeg so the Windows
# installer ships it. Run from the repository root:
#
#   pwsh packaging/windows/fetch-ffmpeg.ps1 [-Url <zip url>] [-Sha256 <hex>]
#
# Pass -Sha256 (or set FFMPEG_SHA256) to pin the exact archive. Without it the
# script prints the hash it saw so a release can record it.
param(
  [string]$Url = $(if ($env:FFMPEG_URL) { $env:FFMPEG_URL } else { "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-7.1.zip" }),
  [string]$Sha256 = $env:FFMPEG_SHA256
)
$ErrorActionPreference = "Stop"
$dest = Join-Path $PSScriptRoot "..\..\apps\desktop\src-tauri\ffmpeg"
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("cf-ffmpeg-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
  $zip = Join-Path $work "ffmpeg.zip"
  Write-Host "Downloading $Url"
  Invoke-WebRequest -Uri $Url -OutFile $zip -UseBasicParsing
  $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
  Write-Host "sha256 $hash"
  if ($Sha256 -and $hash -ne $Sha256.ToLower()) { throw "FFmpeg archive hash mismatch (expected $Sha256)" }
  Expand-Archive -Path $zip -DestinationPath (Join-Path $work "x")
  $bin = Get-ChildItem -Recurse -Path (Join-Path $work "x") -Filter ffmpeg.exe | Select-Object -First 1
  if (-not $bin) { throw "ffmpeg.exe not found in archive" }
  $root = Split-Path (Split-Path $bin.FullName)
  New-Item -ItemType Directory -Force -Path $dest | Out-Null
  Copy-Item (Join-Path $root "bin\ffmpeg.exe"), (Join-Path $root "bin\ffprobe.exe") $dest -Force
  if (Test-Path (Join-Path $root "LICENSE.txt")) { Copy-Item (Join-Path $root "LICENSE.txt") (Join-Path $dest "FFMPEG_LICENSE.txt") -Force }
  "Source: $Url`nsha256: $hash`nFFmpeg is licensed under the GPL; source code: https://ffmpeg.org/download.html and https://github.com/BtbN/FFmpeg-Builds" | Set-Content (Join-Path $dest "FFMPEG_SOURCE.txt")
  & (Join-Path $dest "ffmpeg.exe") -hide_banner -version | Select-Object -First 1
  $filters = & (Join-Path $dest "ffmpeg.exe") -hide_banner -filters
  if (-not ($filters -match "vidstabdetect")) { throw "this FFmpeg build lacks vid.stab" }
  if (-not ($filters -match "zscale")) { throw "this FFmpeg build lacks zscale" }
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
