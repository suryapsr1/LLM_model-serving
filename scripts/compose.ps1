$ErrorActionPreference = 'Stop'
$composeArguments = $args
if ($composeArguments.Count -eq 0) {
    $composeArguments = @('up', '--build', '-d', '--wait', '--wait-timeout', '180')
}

$dockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($dockerCommand) {
    $dockerExecutable = $dockerCommand.Source
} else {
    $dockerCandidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'),
        (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe')
    )
    $dockerExecutable = $dockerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $dockerExecutable) {
    throw 'Docker Desktop is required. Install it, start its Linux engine, then retry.'
}

# Docker invokes its credential helper by name, so the CLI directory must be on PATH.
$previousPath = $env:PATH
$env:PATH = (Split-Path -Parent $dockerExecutable) + [IO.Path]::PathSeparator + $previousPath
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    & $dockerExecutable compose @composeArguments
    $composeExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    $env:PATH = $previousPath
}
exit $composeExitCode
