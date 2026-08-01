param(
    [string]$Python = "python",
    [switch]$InstallDependencies
)

$ErrorActionPreference = "Stop"

if ($InstallDependencies) {
    & $Python -m pip install -r requirements.txt
}

& $Python -m unittest discover -s tests -v
& $Python experiments/run_all.py

Write-Host "PLWM verification completed."
