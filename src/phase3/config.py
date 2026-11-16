from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class TraceConfig:
    """Dataset & strictly causal chronological split configuration."""

    raw_static_dir: Path = ROOT_DIR / "alibaba 2017"
    batch_instance_csv: Path = (
        ROOT_DIR / "data" / "alibaba_v2017" / "alibaba-trace-2017" / "batch_instance.csv"
    )
    schema_csv: Path = ROOT_DIR / "data" / "alibaba_v2017" / "schema.csv"
    processed_dir: Path = ROOT_DIR / "data" / "alibaba_v2017" / "processed"
    porto_cells_parquet: Path = ROOT_DIR / "assets" / "porto_cells.parquet"

    # Verified multi-table overlap [39,600, 81,000) seconds (11.5 hours)
    # where server_usage (300s) and batch_instance (1s) both overlap continuously.
    t_start_s: int = 39600
    t_end_s: int = 81000
    dt_bin_s: int = 30  # 1,380 discrete forecast bins over 11.5h

    # Candidate filter for stable physical servers (1 host = 1 edge node)
    min_nonzero_frac: float = 0.80
    min_mean_tasks_per_min: float = 6.0
    candidate_pool_limit: int = 400
    lookback_L: int = 8  # 8 x 30s = 4 minutes history

    # Strictly chronological proportions (zero future shuffle)
    train_frac: float = 0.60
    val_frac: float = 0.15
    test_frac: float = 0.25


@dataclass(frozen=True)
class TopologyConfig:
    """Edge graph connectivity & physical link assumptions ([Assumption])."""

    n_nodes: int = 6
    k_edge_connectivity: int = 2
    topology_seed: int = 99
    rtt_min_s: float = 0.005  # 5 ms min edge RTT
    rtt_max_s: float = 0.040  # 40 ms max edge RTT
    bw_min_mbps: float = 50.0
    bw_max_mbps: float = 200.0


@dataclass(frozen=True)
class ForecasterConfig:
    """PyTorch LSTM workload forecaster configuration."""

    input_dim: int = 7
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.15
    personalization: Literal["p1_shared", "p2_local_head", "p3_finetune"] = "p2_local_head"
    common_init: bool = True  # Shared initial weights theta_0 keep gossip nodes in compatible basins
    calibrate_after_merge: bool = False  # Optional extra post-merge gradient calibration step
    local_adapt_on_causal_surprise: bool = True  # Matched local burst adaptation across all comparators
    pretrain_epochs: int = 40
    pretrain_batch_size: int = 64
    lr: float = 2e-3
    online_lr: float = 5e-4
    online_steps_per_trigger: int = 2
    online_buffer_len: int = 32
    head_bytes_header: int = 24


@dataclass(frozen=True)
class APSMConfig:
    """Online semantic surprise & gating configuration (APSM-W)."""

    mode: Literal["off", "h1_residual", "val_delta"] = "h1_residual"
    stat: Literal["std", "mad", "ewma"] = "std"
    k: float = 1.5
    window: int = 50
    min_obs: int = 10
    heartbeat: int = 5
    retrain_on_trigger: bool = True
    periodic_online_retrain_interval: int = 4  # Fallback online fit steps when not surprise-triggered
    surprise_clip: float = 2.0


@dataclass(frozen=True)
class GossipConfig:
    """Equal-cadence decentralized Gossip Learning protocol."""

    cadence_bins: int = 2  # Every 2 x 30s = 60s gossip decision check (EQUAL across GL & APSM!)
    target_probability: float = 1.0  # 1.0 = all 1-hop neighbors per active send
    model_tx_delay_s: float = 0.05  # Actual ~54KB / 100Mbps ~ 4.3ms (< 1 bin; deterministic arrival < next bin)
    age_decay: float = 0.05
    self_mix_alpha: float = 0.70  # Self-anchor blend floor during age-weighted P2P merges


@dataclass(frozen=True)
class SimConfig:
    """Deterministic Task-Queueing DES & Offloading configuration (Stage 4-6)."""

    dt_slot_s: float = 1.0
    node_cpu_cap: float = 20.0  # Calibrated edge-tier cores/s (B0 ~24.5% DMR under real trace bursts)
    node_mem_cap: float = 1.0
    q_max: int = 48
    bg_cpu_cap: float = 0.75
    bg_mem_cap: float = 0.55
    task_mem_scale: float = 0.45
    task_nom_cpu_mult: float = 2.0
    deadline_slack_kappa: float = 2.2
    deadline_base_slack_s: float = 3.0
    payload_bytes_min: int = 50_000
    payload_bytes_max: int = 1_000_000
    telemetry_interval_s: float = 15.0
    telemetry_beacon_bytes: int = 32
    ack_bytes: int = 16
    task_header_bytes: int = 32
    k_max_neighbors: int = 8
    info_age_max_s: float = 60.0
    # Normalization constants for permutation-invariant MARL obs
    obs_w_scale: float = 100.0
    obs_bytes_scale: float = 1_000_000.0
    obs_dl_scale: float = 120.0
    # Reward weights (bounded in [-(w_drop + ...), 0]; w_drop >> w_lat stops drop hacking)
    reward_w_lat: float = 1.0
    reward_w_drop: float = 3.0
    reward_w_net: float = 0.15
    reward_w_imb: float = 0.20
    drop_penalty_lat_mult: float = 2.0


@dataclass(frozen=True)
class MARLConfig:
    """Parameter-shared Masked MAPPO / IPPO / Double Dueling MADQN configuration (Stage 5-6)."""

    hidden_dim: int = 128
    gamma: float = 0.40
    gae_lambda: float = 0.90
    clip_eps: float = 0.20
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    entropy_coef: float = 0.010
    entropy_min: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 8
    minibatch_size: int = 256
    n_train_episodes: int = 48
    train_window_bins: int = 120
    eval_every: int = 6
    # Disjoint pre-test Wave-1 burst validation slice [380, 488) (zero overlap with RL train or held-out test)
    burst_val_start_bin: int = 380
    burst_val_end_bin: int = 488
    # MADQN comparator hyperparameters
    dqn_lr: float = 3e-4
    dqn_buffer_size: int = 60_000
    dqn_batch_size: int = 256
    dqn_tau: float = 0.005
    dqn_eps_start: float = 0.30
    dqn_eps_end: float = 0.01
    dqn_eps_decay_episodes: int = 28
    dqn_updates_per_episode: int = 48


@dataclass(frozen=True)
class Phase3Config:
    """Top-level reproducible configuration."""

    tier_name: str = "tier1_dev_6n"
    seed: int = 42
    device: str = "cpu"
    trace: TraceConfig = field(default_factory=TraceConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    apsm: APSMConfig = field(default_factory=APSMConfig)
    gossip: GossipConfig = field(default_factory=GossipConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    marl: MARLConfig = field(default_factory=MARLConfig)


def dev_tier1_6n(seed: int = 42) -> Phase3Config:
    """Tier 1 local development & verification preset (6 nodes, K=2 sparse mesh)."""
    return Phase3Config(
        tier_name="tier1_dev_6n",
        seed=seed,
        topology=TopologyConfig(n_nodes=6, k_edge_connectivity=2),
    )


def eval_tier2_10n(seed: int = 42, device: str = "cuda") -> Phase3Config:
    """Tier 2 primary evaluation preset (10 nodes, K=3 sparse mesh)."""
    return Phase3Config(
        tier_name="tier2_eval_10n",
        seed=seed,
        device=device,
        topology=TopologyConfig(n_nodes=10, k_edge_connectivity=3),
    )


def eval_tier2_50n(seed: int = 42, device: str = "cuda") -> Phase3Config:
    """Tier 2 large-scale evaluation preset (50 nodes, K=3 sparse mesh)."""
    return Phase3Config(
        tier_name="tier2_eval_50n",
        seed=seed,
        device=device,
        topology=TopologyConfig(n_nodes=50, k_edge_connectivity=3),
    )


def eval_tier2_100n(seed: int = 42, device: str = "cuda") -> Phase3Config:
    """Tier 2 zero-shot transfer scale preset (100 nodes, K=3 sparse mesh)."""
    return Phase3Config(
        tier_name="tier2_transfer_100n",
        seed=seed,
        device=device,
        topology=TopologyConfig(n_nodes=100, k_edge_connectivity=3),
    )

