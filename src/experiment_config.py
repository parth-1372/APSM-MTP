"""
experiment_config.py
====================
Central configuration file for all scalability experiments.
Change these parameters to update the entire pipeline (network generation, dataset generation, and model simulation).
"""

# ──────────────────────────────────────────────────────────────────────
# 1. Global Setup
# ──────────────────────────────────────────────────────────────────────
# The random seed used for generating networks and initial datasets. 
# Keeping this constant ensures the same map topography is generated.
GLOBAL_SEED = 99

# The number of simulated networks/datasets to generate in advance.
# (Usually set to 10 so you have a pool of topographies to test on).
N_SIMULATIONS = 10


# ──────────────────────────────────────────────────────────────────────
# 2. Network & Dataset Topography
# ──────────────────────────────────────────────────────────────────────
# The number of geographical nodes (towers) to sample for the network.
N_NODES = 50

# The number of nearest neighbours each node connects to in the graph.
K_EDGE_CONNECTIVITY = 3

# If greater than 0, assigns a "small dataset" to this many nodes (data imbalance).
N_SMALL_DATASET_NODES = 0


# ──────────────────────────────────────────────────────────────────────
# 3. Model Training & Simulation Configuration
# ──────────────────────────────────────────────────────────────────────
# Number of independent runs to perform (to average out geographical luck).
# Replaces the hardcoded loop in run_all_comparisons.
N_SEEDS = 1

# Set to True to run seeds in parallel across CPU/GPU, or False to run sequentially (saves RAM).
PARALLEL_SEEDS = False

# Number of total gossip rounds each node will execute.
# Must be greater than SEMANTIC_WINDOW (50) for the filter to engage.
FIXED_UPDATES = 100

# Number of training epochs to run on the local data per gossip update.
EPOCHS_PER_UPDATE = 3

# Limit the maximum number of samples per node to speed up sanity checks.
# Set to an integer (e.g., 500) for testing, or None for the FULL DATA.
MAX_SAMPLES = None


# ──────────────────────────────────────────────────────────────────────
# 4. APSM Predictive-Semantic Filter Configuration
# ──────────────────────────────────────────────────────────────────────
# The noise-band multiplier τ(t) = K · σ(ε). 
# A value of 2.0 captures roughly the 95% confidence interval of surprise.
SEMANTIC_K = 2.0

# The sliding window size N used to calculate the rolling standard deviation σ(ε).
SEMANTIC_WINDOW = 50

# The maximum number of times a node is allowed to suppress transmission in a row.
# Forces a 'heartbeat' send to prevent network deadlock.
SEMANTIC_HEARTBEAT = 5
