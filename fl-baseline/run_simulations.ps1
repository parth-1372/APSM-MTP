param (
    [int]$networkScenarios = 10,
    [int]$simulationRuns = 10,
    [string]$experimentsDir = "experiments",
    [double]$fractionFit = 0.5,
    [int]$startScenario = 0,
    [int]$startRun = 0
)

# Run the simulations for the federated learning baseline on Windows PowerShell
Write-Host "Starting Flower FL Simulations on Windows..."
for ($i = $startScenario; $i -lt $networkScenarios; $i++) {
    for ($j = $startRun; $j -lt $simulationRuns; $j++) {
        $runConfig = "network-scenario=$i simulation-run=$j experiments-dir='$experimentsDir' fraction-fit=$fractionFit"
        Write-Host "Running: flwr run . --run-config `"$runConfig`""
        flwr run . --run-config $runConfig
    }
    $startRun = 0
}
Write-Host "FL Simulations completed."
