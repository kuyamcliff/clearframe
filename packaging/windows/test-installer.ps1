# Installer lifecycle test for the NSIS setup. Run on a disposable Windows
# machine or CI runner as Administrator (the installer is per-machine):
#
#   pwsh packaging/windows/test-installer.ps1 -Installer <ClipFX Enhance_x.y.z_x64-setup.exe>
#
# Covers: silent install, self-test of the installed app (FFmpeg found, user
# folders writable), upgrade over an existing install keeping user data,
# uninstall keeping user data, and install into a custom folder containing
# spaces and non-ASCII characters.
#
# It writes only a marker file inside %LOCALAPPDATA%\ClipFX Enhance and removes it
# at the end; it refuses to run if that folder already holds real user data
# unless -AllowExistingData is given.
param(
  [Parameter(Mandatory = $true)][string]$Installer,
  [switch]$AllowExistingData
)
$ErrorActionPreference = "Stop"
$Installer = (Resolve-Path $Installer).Path
$data = Join-Path $env:LOCALAPPDATA "ClipFX Enhance"
$defaultDir = Join-Path $env:ProgramFiles "ClipFX Enhance"
$customDir = Join-Path $env:ProgramFiles "ClipFX Enhance Test Ünïcode Dir"
$results = [System.Collections.Generic.List[string]]::new()

function Step($name) { Write-Host "`n=== $name ===" }
function Pass($name) { $results.Add("PASS  $name"); Write-Host "PASS  $name" -ForegroundColor Green }
function Fail($name) { $results.Add("FAIL  $name"); throw "FAIL  $name" }

if ((Test-Path (Join-Path $data "clipfx.sqlite3")) -and -not $AllowExistingData) {
  throw "$data already contains ClipFX Enhance data. Run this test on a clean machine or pass -AllowExistingData."
}

function Install([string]$dir) {
  $argList = @("/S")
  if ($dir) { $argList += "/D=$dir" }   # /D must be last and unquoted
  $p = Start-Process -FilePath $Installer -ArgumentList $argList -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "installer exited with $($p.ExitCode)" }
}

function Uninstall([string]$dir) {
  $u = Join-Path $dir "uninstall.exe"
  if (-not (Test-Path $u)) { throw "no uninstaller at $u" }
  # _?= makes NSIS run in place and wait; it then cannot delete itself, so remove it after.
  $p = Start-Process -FilePath $u -ArgumentList @("/S", "_?=$dir") -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "uninstaller exited with $($p.ExitCode)" }
  Remove-Item -Force $u -ErrorAction SilentlyContinue
  Remove-Item -Force $dir -ErrorAction SilentlyContinue
}

function SelfTest([string]$dir) {
  $exe = Join-Path $dir "ClipFXEnhance.exe"
  if (-not (Test-Path $exe)) { throw "ClipFXEnhance.exe missing in $dir" }
  $out = Join-Path ([System.IO.Path]::GetTempPath()) ("cf-selftest-" + [guid]::NewGuid() + ".json")
  $p = Start-Process -FilePath $exe -ArgumentList @("--self-test", "`"$out`"") -Wait -PassThru
  $json = Get-Content $out -Raw | ConvertFrom-Json
  Remove-Item $out -Force
  if ($p.ExitCode -ne 0 -or -not $json.ok) { throw "self-test failed: $($json | ConvertTo-Json -Depth 5)" }
  return $json
}

try {
  Step "fresh silent install"
  Install ""
  if (-not (Test-Path (Join-Path $defaultDir "ClipFXEnhance.exe"))) { Fail "installed into Program Files" }
  Pass "installed into $defaultDir"
  $r = SelfTest $defaultDir
  if (-not $r.ffmpeg) { Fail "bundled FFmpeg found ($($r.ffmpeg_error))" }
  if ($r.ffmpeg.source -ne "bundled") { Fail "FFmpeg comes from the install folder (got $($r.ffmpeg.source))" }
  if (-not ($r.ffmpeg.vidstab -and $r.ffmpeg.zscale)) { Fail "FFmpeg has vid.stab and zscale" }
  Pass "self-test: FFmpeg $($r.ffmpeg.version) bundled, folders writable"
  if (-not ($r.paths.models_dir -like "$data*")) { Fail "models live under %LOCALAPPDATA%\ClipFX Enhance (got $($r.paths.models_dir))" }
  Pass "per-user data in $($r.paths.data_dir)"
  if (Get-ChildItem -Path $defaultDir -Recurse -Include *.db, models, cache -ErrorAction SilentlyContinue) { Fail "no user data inside Program Files" }
  Pass "no user data inside Program Files"

  $marker = Join-Path $r.paths.models_dir "installer-test-marker.txt"
  New-Item -ItemType Directory -Force -Path (Split-Path $marker) | Out-Null
  Set-Content -Path $marker -Value "keep me"

  Step "upgrade (reinstall over existing)"
  Install ""
  SelfTest $defaultDir | Out-Null
  if (-not (Test-Path $marker)) { Fail "upgrade keeps user data" }
  Pass "upgrade keeps user data"

  Step "silent uninstall keeps user data"
  Uninstall $defaultDir
  if (Test-Path (Join-Path $defaultDir "ClipFXEnhance.exe")) { Fail "program files removed" }
  if (-not (Test-Path $marker)) { Fail "uninstall keeps user data" }
  Pass "uninstall removed the program and kept user data"

  Step "install into a custom folder with spaces and non-ASCII characters"
  Install $customDir
  $r = SelfTest $customDir
  if (-not (Test-Path $marker)) { Fail "reinstall finds existing data" }
  Pass "custom folder install works and reuses existing data"
  Uninstall $customDir
  if (Test-Path (Join-Path $customDir "ClipFXEnhance.exe")) { Fail "custom folder uninstall" }
  Pass "custom folder uninstall"
}
finally {
  if (Test-Path $marker) { Remove-Item -Force $marker }
  Write-Host "`nSummary:"
  $results | ForEach-Object { Write-Host $_ }
}
