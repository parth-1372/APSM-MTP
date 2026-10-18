import subprocess
import sys
import typer

app = typer.Typer()


def _run_flwr(run_config: str):
    """Run Flower simulation cross-platform via subprocess."""
    cmd = [sys.executable, "-m", "flwr.cli", "run", ".", "--run-config", run_config]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        # Fallback to direct 'flwr run' command if module invocation differs
        cmd_fallback = f"flwr run . --run-config \"{run_config}\""
        print(f"Retrying with fallback: {cmd_fallback}")
        subprocess.run(cmd_fallback, shell=True, check=True)


@app.command("sampling")
def run_sampling_exp(
    scenarios: int = 10,
    runs: int = 10,
    start_from_scenario: int = 0,
    start_from_run: int = 0,
    experiments_dir: str = "experiments",
    sampling_percentage: float = 0.5,
):
    for i in range(start_from_scenario, scenarios):
        for j in range(start_from_run, runs):
            cfg = f"network-scenario={i} simulation-run={j} experiments-dir='{experiments_dir}' fraction-fit={sampling_percentage}"
            _run_flwr(cfg)
        start_from_run = 0


@app.command("epochs-updates")
def run_epochs_updates_exp(
    scenarios: int = 10,
    runs: int = 10,
    start_from_scenario: int = 0,
    start_from_run: int = 0,
    epochs: int = 1,
    updates: int = 100,
):
    experiments_dir = f"experiments/epochs-updates/{epochs}-{updates}"
    for i in range(start_from_scenario, scenarios):
        for j in range(start_from_run, runs):
            cfg = f"network-scenario={i} simulation-run={j} experiments-dir='{experiments_dir}' local-epochs={epochs} num-server-rounds={updates}"
            _run_flwr(cfg)
        start_from_run = 0


if __name__ == "__main__":
    app()