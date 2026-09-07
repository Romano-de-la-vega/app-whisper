[CmdletBinding()]
param([string]$IsccPath = "", [switch]$RequireInstaller)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') { throw 'Windows 10/11 x64 required.' }

function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program exited with code $LASTEXITCODE" }
}

function Remove-BuildDirectory([string]$RelativePath) {
    $target = [IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
    if ($target -ne (Join-Path $projectRoot 'build') -and $target -ne (Join-Path $projectRoot 'dist')) {
        throw "Refusing to delete unexpected path: $target"
    }
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Force -Recurse }
}

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
$uv = if ($uvCommand) { $uvCommand.Source } else { Join-Path $projectRoot 'venv\Scripts\uv.exe' }
if (-not (Test-Path -LiteralPath $uv)) {
    $bootstrap = Get-Command python -ErrorAction Stop
    Invoke-Checked $bootstrap.Source @('-m', 'pip', 'install', 'uv>=0.8,<1')
    $uv = (& $bootstrap.Source -c "import pathlib,uv; print(pathlib.Path(uv.__file__).parent / 'uv.exe')").Trim()
    if (-not (Test-Path -LiteralPath $uv)) { throw 'uv not found. Install uv and rerun this script.' }
}
Invoke-Checked $uv @('sync', '--locked', '--group', 'dev', '--group', 'build', '--python', '3.11')
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
Invoke-Checked $python @('-c', 'import sys; assert sys.maxsize>2**32; assert sys.version_info[:2]==(3,11)')
Invoke-Checked $python @('-m', 'ruff', 'check', '.')
$previousQt = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = 'offscreen'
try {
    Invoke-Checked $python @('-m', 'pytest', '-q')
    Invoke-Checked $python @('-m', 'compileall', '-q', 'src', 'tests', 'packaging')
    Remove-BuildDirectory 'build'
    Remove-BuildDirectory 'dist'
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'build') | Out-Null
    Invoke-Checked $python @('-m', 'transcripteur_whisper', '--smoke-test', '--smoke-report', 'build/source-smoke.json')
    Invoke-Checked $python @('-m', 'PyInstaller', '--noconfirm', '--clean', 'packaging/TranscripteurWhisper.spec')
    $bundle = Join-Path $projectRoot 'dist\TranscripteurWhisper'
    $executable = Join-Path $bundle 'TranscripteurWhisper.exe'
    foreach ($required in @('TranscripteurWhisper.exe', '_internal\python311.dll', '_internal\transcripteur_whisper\assets\icon.ico', '_internal\PySide6\Qt6Multimedia.dll', '_internal\PySide6\plugins\platforms\qwindows.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $bundle $required))) { throw "Missing essential bundle file: $required" }
    }
    $smokeReport = Join-Path $projectRoot 'build\bundle-smoke.json'
    $smokeError = Join-Path $projectRoot 'build\bundle-smoke-stderr.log'
    $smoke = Start-Process -FilePath $executable -ArgumentList @('--smoke-test', '--smoke-report', ('"' + $smokeReport + '"')) -PassThru -WindowStyle Hidden -RedirectStandardError $smokeError
    $null = $smoke.Handle
    if (-not $smoke.WaitForExit(120000)) { $smoke.Kill(); throw 'Bundle smoke test timed out.' }
    if ($smoke.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $smokeReport)) { throw 'Bundle smoke test failed. Inspect user logs.' }
    if (-not (Get-Content -LiteralPath $smokeReport -Raw | ConvertFrom-Json).ok) { throw 'Bundle smoke report failed.' }
} finally { $env:QT_QPA_PLATFORM = $previousQt }

$version = (& $python -c 'from transcripteur_whisper import __version__; print(__version__)').Trim()
if (-not $IsccPath) {
    $isccCommand = Get-Command ISCC -ErrorAction SilentlyContinue
    if ($isccCommand) { $IsccPath = $isccCommand.Source }
    else {
        foreach ($candidate in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
            if (Test-Path -LiteralPath $candidate) { $IsccPath = $candidate; break }
        }
    }
}
$installer = $null
if ($IsccPath) {
    Invoke-Checked $IsccPath @("/DAppVersion=$version", 'packaging/installer.iss')
    $installer = Join-Path $projectRoot "dist\installer\TranscripteurWhisper-Setup-$version.exe"
    if (-not (Test-Path -LiteralPath $installer)) { throw 'Installer missing after compilation.' }
} elseif ($RequireInstaller) { throw 'Inno Setup 6 not found. Supply -IsccPath or install it.' }
else { Write-Warning 'Inno Setup 6 not found: validated application bundle only; no installer produced.' }

$hashFile = Join-Path $projectRoot 'dist\SHA256SUMS.txt'
$hashLines = foreach ($file in (Get-ChildItem -LiteralPath $bundle -Recurse -File)) {
    $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
    '{0}  {1}' -f $hash.Hash.ToLowerInvariant(), $file.FullName.Substring((Join-Path $projectRoot 'dist').Length + 1)
}
if ($installer) { $hashLines += ('{0}  installer\{1}' -f (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant(), (Split-Path $installer -Leaf)) }
$hashLines | Set-Content -LiteralPath $hashFile -Encoding UTF8
Write-Host "Build successful`nApplication:`n$executable"
if ($installer) { Write-Host "Installer:`n$installer" }
Write-Host "SHA256 manifest:`n$hashFile"
Get-FileHash -LiteralPath $executable -Algorithm SHA256 | Format-List
if ($installer) { Get-FileHash -LiteralPath $installer -Algorithm SHA256 | Format-List }
