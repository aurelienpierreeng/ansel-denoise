# One-shot setup of everything a Windows contributor needs to harvest and
# pack training shards for the Ansel raw denoiser. PowerShell twin of
# setup_contributor.sh — no git required: if the repository is not present,
# it is downloaded as a ZIP.
#
# Run it from PowerShell (Win+X -> "Terminal" or "Windows PowerShell"):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_contributor.ps1
#
# Requires Python >= 3.10 from https://python.org (check "Add python.exe to
# PATH" during install) or:  winget install Python.Python.3.12

$ErrorActionPreference = "Stop"
$RepoZip = "https://github.com/aurelienpierreeng/ansel-denoise/archive/refs/heads/master.zip"

# --- find a Python >= 3.10 -------------------------------------------------
$py = $null
foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
    $exe = Get-Command $cand[0] -ErrorAction SilentlyContinue
    if (-not $exe) { continue }
    & $cand[0] $cand[1..($cand.Count)] -c "import sys; sys.exit(sys.version_info < (3, 10))" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = $cand; break }
}
if (-not $py) {
    Write-Host "ERROR: Python >= 3.10 not found." -ForegroundColor Red
    Write-Host "Install it with:  winget install Python.Python.3.12"
    Write-Host "or from https://python.org (check 'Add python.exe to PATH'), then rerun this script."
    exit 1
}
$pyv = & $py[0] $py[1..($py.Count)] --version
Write-Host "using $($py -join ' ') ($pyv)"

# --- find or download the repository ---------------------------------------
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $scriptDir
if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    if (Test-Path ".\ansel-denoise\pyproject.toml") {
        $repo = (Resolve-Path ".\ansel-denoise").Path
    } else {
        Write-Host "downloading the repository (no git needed)..."
        $zip = Join-Path $env:TEMP "ansel-denoise.zip"
        Invoke-WebRequest -Uri $RepoZip -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath "." -Force
        Remove-Item $zip
        if (Test-Path ".\ansel-denoise") { Remove-Item -Recurse -Force ".\ansel-denoise" }
        Rename-Item "ansel-denoise-master" "ansel-denoise"
        $repo = (Resolve-Path ".\ansel-denoise").Path
    }
}
Write-Host "repo: $repo"

# --- install the Python tooling (numpy + rawpy; both have Windows wheels) --
& $py[0] $py[1..($py.Count)] -m pip install --user -e "$repo[harvest]"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed." -ForegroundColor Red
    exit 1
}

# --- smoke check -----------------------------------------------------------
& $py[0] $py[1..($py.Count)] -c @"
import numpy, rawpy
import ansel_denoise.harvest_library, ansel_denoise.validate_shards
from ansel_denoise.harvest_library import DEFAULT_DB
print('tooling OK: numpy', numpy.__version__, '| rawpy', rawpy.__version__)
print('Ansel library expected at:', DEFAULT_DB, '| exists:', DEFAULT_DB.is_file())
"@
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Setup complete. Next steps (details in $repo\CONTRIBUTING.md):"
Write-Host "  1. In Ansel: select images (base ISO, content you agree to publish),"
Write-Host "     then File > Export image list... > Save as file..."
Write-Host "  2. cd `"$repo`""
Write-Host "     py -m ansel_denoise.harvest_library --paths-file ansel-image-files.txt --out shards\mine"
Write-Host "  3. py scripts\pack_contribution.py shards\mine --handle your-name"
Write-Host "  4. Upload the bundle, then:"
Write-Host "     powershell -ExecutionPolicy Bypass -File scripts\submit_contribution.ps1 -Bundle <bundle.tar.gz> -Url <link>"
