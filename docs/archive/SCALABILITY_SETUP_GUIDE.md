# Scalability Experiments Setup Guide

This guide contains the exact steps and commands required to set up the repository on a fresh, high-power machine and run the Scalability Experiments (from 10 up to 100 nodes).

## 1. Initial Setup & Installation

Open your terminal and run the following commands to download the code and install all dependencies:

```bash
# Clone the repository
git clone https://github.com/parth-1372/APSM-MTP.git
cd APSM-MTP

# Install Poetry and core dependencies
pip install -q poetry
poetry config virtualenvs.create false
poetry install --no-interaction

# Install missing data processing libraries
pip install category-encoders fastparquet pyarrow
```

## 2. Prepare the Raw Dataset

Because the Porto Taxi Dataset (`taxi_porto.csv`) is 1.9 GB, it is too large for GitHub. On your new machine, you must manually provide this data:

1. Copy your existing `assets/` folder from your current machine to the root of the new project directory. (It must contain `taxi_porto.csv` and `porto_cells.parquet`).
2. Alternatively, download the dataset from Kaggle (`crawford/taxi-trajectory`), rename the main CSV to `taxi_porto.csv`, and place it inside the `assets/` folder.

```bash
# Ensure the data output directories exist
mkdir -p data/datasets
mkdir -p data/networks

# Re-create the bounding box file required by the network generator
mkdir -p assets
echo "(-8.6338, -8.5862, 41.1369, 41.1690)" > assets/BBox_Porto.txt
```

## 3. Workflow to Run an Experiment Scale (e.g., 50 Nodes)

To run a scalability test for a specific number of nodes (for example, `50`), follow these three steps:

### A. Generate the Network Graph
1. Open `src/1_network_generation.py` in your code editor.
2. Change the variable near the top to `n_nodes = 50`.
3. Run the script from the terminal:
```bash
PYTHONPATH=src python src/1_network_generation.py
```

### B. Generate the Node Datasets
1. Open `src/2_dataset_generation.py` in your editor.
2. Change the variable at the top to `n_nodes = 50`.
3. Run the script from the terminal:
```bash
PYTHONPATH=src python src/2_dataset_generation.py
```

### C. Run the Simulation
1. Open `src/run_all_comparisons.py` in your editor.
2. Change the variable at the top to `N_NODES = 50`.
3. Run the simulation from the terminal:
```bash
PYTHONPATH=src python src/run_all_comparisons.py
```
*Note: This will safely save all your results and metrics into a dedicated folder: `results/phase2_full_50n/`.*

## 4. Repeat and Plot

Once you have completed the workflow for all the scales you want to test (e.g., 10, 20, 50, 100 nodes), you can generate your final thesis graphs instantly without re-running any heavy simulations.

Simply run the plotter tool:
```bash
PYTHONPATH=src python src/plot_scalability.py
```

This will scan your `results/` folder and generate two master plots combining all scales:
- `scalability_bandwidth_plot.png`
- `scalability_accuracy_plot.png`
