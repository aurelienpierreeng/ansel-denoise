# Submit a contribution bundle as a pull request from Windows — no git
# knowledge needed. PowerShell twin of submit_contribution.sh: the GitHub CLI
# signs you in through your browser, forks the repository and opens the pull
# request; the PR carries only a small metadata file, never the images.
#
# Requirements (one-time):  winget install Git.Git GitHub.cli
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\submit_contribution.ps1 `
#       -Bundle ansel-denoise-contrib-you-20260719.tar.gz -Url https://...

param(
    [Parameter(Mandatory = $true)][string]$Bundle,
    [Parameter(Mandatory = $true)][string]$Url
)
$ErrorActionPreference = "Stop"
$Repo = if ($env:ANSEL_DENOISE_REPO) { $env:ANSEL_DENOISE_REPO } else { "aurelienpierreeng/ansel-denoise" }

if (-not (Test-Path $Bundle)) { Write-Host "no such file: $Bundle" -ForegroundColor Red; exit 1 }
foreach ($tool in @("gh", "git", "tar")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$tool' is required." -ForegroundColor Red
        Write-Host "Install with:  winget install Git.Git GitHub.cli"
        Write-Host "No winget (common on Windows 10)? Download the installers directly:"
        Write-Host "  Git for Windows: https://git-scm.com/download/win"
        Write-Host "  GitHub CLI:      https://cli.github.com (Download for Windows)"
        Write-Host "then close and reopen PowerShell. Or skip the installs entirely and"
        Write-Host "open a 'Shard contribution' issue with your link and checksum:"
        Write-Host "  https://github.com/aurelienpierreeng/ansel-denoise/issues/new/choose"
        exit 1
    }
}
# sign in through the browser if needed — the only 'account setup' there is
gh auth status 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { gh auth login --web }

$sha = (Get-FileHash $Bundle -Algorithm SHA256).Hash.ToLower()
$work = Join-Path $env:TEMP ("ansel-contrib-" + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path "$work\x" -Force | Out-Null

# --- pending metadata file from the bundle's own manifest ------------------
tar xzf $Bundle -C "$work\x"
$manifestFile = Get-ChildItem -Path "$work\x" -Recurse -Filter "contribution-manifest.json" | Select-Object -First 1
if (-not $manifestFile) {
    Write-Host "not a contribution bundle (no manifest) - run pack_contribution.py first" -ForegroundColor Red
    exit 1
}
$manifest = Get-Content $manifestFile.FullName -Raw | ConvertFrom-Json
$handle = $manifest.handle
$pending = [ordered]@{}
foreach ($p in $manifest.PSObject.Properties) {
    if ($p.Name -ne "files") { $pending[$p.Name] = $p.Value }  # per-file hashes stay in the bundle
}
$pending["url"] = $Url
$pending["bundle_sha256"] = $sha
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmm")
$name = "$handle-$stamp"
$pendingJson = Join-Path $work "pending.json"
$pending | ConvertTo-Json | Set-Content -Path $pendingJson -Encoding UTF8

# --- fork, branch, commit, pull request — all through gh -------------------
Write-Host "forking $Repo and opening the pull request..."
Push-Location $work
gh repo fork $Repo --clone -- --depth 1 --quiet
if ($LASTEXITCODE -ne 0) { Pop-Location; exit 1 }
$clone = Join-Path $work (Split-Path $Repo -Leaf)
New-Item -ItemType Directory -Path "$clone\contrib\pending" -Force | Out-Null
Copy-Item $pendingJson "$clone\contrib\pending\$name.json"

Set-Location $clone
git checkout -q -b "shards/$name"
git add "contrib/pending/$name.json"
# identity fallback so first-time users without a git config can commit
git -c user.name="$handle" -c user.email="$handle@users.noreply.github.com" `
    commit -q -m "Shard contribution from $handle"
git push -q -u origin "shards/$name"

$body = @"
Shard contribution bundle (packed by ``pack_contribution.py``, see CONTRIBUTING.md).

- download: $Url
- sha256: ``$sha``

By opening this pull request I confirm the grant recorded in the metadata
file: I own the rights to these photographs and license the tiles under the
[Ansel Training Data License 1.1](../blob/master/LICENSE-DATA.md) - usable
with the ansel-denoise training stack to audit/benchmark or to train
denoising models (resulting weights unrestricted); any stack able to learn
anything else (style, generative AI) is forbidden.

The maintainer ingests with:
``./scripts/collect_contribution.sh contrib/pending/$name.json --source <this PR URL>``
"@
$bodyFile = Join-Path $work "pr-body.md"
$body | Set-Content -Path $bodyFile -Encoding UTF8
gh pr create --repo $Repo --base master --title "[shards] contribution from $handle" --body-file $bodyFile
Pop-Location

Write-Host "done - the maintainer will fetch and validate the bundle from your link."
Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
