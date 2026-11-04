$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$HostAddress = if ($env:GEM_GUI_HOST) { $env:GEM_GUI_HOST } else { "127.0.0.1" }
$Port = if ($env:GEM_GUI_PORT) { $env:GEM_GUI_PORT } else { "8080" }
$Workspace = if ($env:GEM_GUI_WORKSPACE) { $env:GEM_GUI_WORKSPACE } else { $Root }

Write-Host "Starting GEM Learning Engine GUI at http://${HostAddress}:${Port}"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv run gem-learning-engine-gui --workspace $Workspace --host $HostAddress --port $Port
} elseif (Test-Path "$Root\.venv\Scripts\gem-learning-engine-gui.exe") {
    & "$Root\.venv\Scripts\gem-learning-engine-gui.exe" --workspace $Workspace --host $HostAddress --port $Port
} else {
    throw "The project is not installed. Run .\scripts\setup.ps1 first."
}
