import numpy as np
import pandas as pd

from phase3.config import dev_tier1_6n
from phase3.data.alibaba_loader import (
    build_or_load_curated_trace,
    make_supervised_windows,
    verify_schema_csv,
)
from phase3.data.topology import build_edge_topology, summarize_topology


def test_schema_verification():
    cfg = dev_tier1_6n()
    schema = verify_schema_csv(cfg.trace.schema_csv)
    assert "batch_instance.csv" in schema
    assert schema["batch_instance.csv"][8] == "maximum real cpu number"
    assert schema["batch_instance.csv"][9] == "average real cpu number"


def test_topology_tier1_sparse_partial_mesh():
    cfg = dev_tier1_6n()
    topo = build_edge_topology(cfg.trace.porto_cells_parquet, cfg.topology)
    summary = summarize_topology(topo)
    assert topo.n_nodes == 6
    # Verify N=6, K=2 yields non-complete partial 1-hop mesh (some peers require >1 hop)
    assert summary["is_fully_connected"] is False
    assert min(len(n) for n in topo.neighbors) >= 2
    assert summary["rtt_ms_min_max"][0] >= 4.9
    assert summary["rtt_ms_min_max"][1] <= 40.1
    assert summary["bw_mbps_min_max"][0] >= 50.0


def test_curated_6n_dataset_causality_and_non_leakage():
    cfg = dev_tier1_6n()
    ds = build_or_load_curated_trace(cfg.trace, n_nodes=cfg.topology.n_nodes)

    assert ds.n_nodes == 6
    assert ds.n_bins == 1380
    assert ds.features.shape == (6, 1380, 7)
    assert ds.targets.shape == (6, 1380)
    assert np.isfinite(ds.features).all()
    assert np.isfinite(ds.targets).all()
    assert np.all(ds.targets >= 0.0) and np.all(ds.targets <= 1.5)
    assert ds.train_end_bin == 828
    assert ds.val_end_bin == 1035

    # Verify y_scale equals exactly 99th percentile of train split ONLY (zero val/test leakage)
    for i in range(ds.n_nodes):
        expected_scale = max(1e-3, float(np.percentile(ds.raw_cpu_rate[i, : ds.train_end_bin], 99)))
        assert np.isclose(ds.y_scales[i], expected_scale, rtol=1e-5)

    # Verify strict causal supervised window indexing (window ends at t-1, target is t)
    X_tr, y_tr = make_supervised_windows(ds.features[0], ds.targets[0], ds.lookback_L, 0, ds.train_end_bin)
    X_te, y_te = make_supervised_windows(ds.features[0], ds.targets[0], ds.lookback_L, ds.val_end_bin, ds.n_bins)
    assert len(X_tr) == ds.train_end_bin - ds.lookback_L
    assert len(X_te) == ds.n_bins - ds.val_end_bin
    # Last timestep of X_tr[0] equals features[lookback_L - 1], target equals targets[lookback_L]
    assert np.allclose(X_tr[0, -1], ds.features[0, ds.lookback_L - 1])
    assert np.isclose(y_tr[0], ds.targets[0, ds.lookback_L])
    # First test window uses strictly <= val_end_bin - 1 features to predict val_end_bin
    assert np.allclose(X_te[0, -1], ds.features[0, ds.val_end_bin - 1])
    assert np.isclose(y_te[0], ds.targets[0, ds.val_end_bin])

    # Verify extracted Parquet tasks align with selected_machine_ids & overlap timestamps
    tasks_df = pd.read_parquet(ds.task_events_path)
    assert len(tasks_df) > 40_000
    assert set(tasks_df["machine_id"].unique()).issubset(set(ds.selected_machine_ids.tolist()))
    assert tasks_df["start_ts"].min() >= cfg.trace.t_start_s
    assert tasks_df["start_ts"].max() < cfg.trace.t_end_s
    assert (tasks_df["duration_s"] >= 1.0).all()

