import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase3.config import TraceConfig


REQUIRED_SCHEMA_COLS = {
    "batch_instance.csv": 12,
    "batch_task.csv": 8,
    "server_event.csv": 7,
    "server_usage.csv": 8,
}


@dataclass
class CuratedTraceDataset:
    """Strictly causal trace bundle for Stage 3 forecasting and Stage 4-6 MARL."""

    n_nodes: int
    n_bins: int
    dt_bin_s: int
    t_start_s: int
    t_end_s: int
    train_end_bin: int
    val_end_bin: int
    lookback_L: int
    selected_machine_ids: np.ndarray  # (N,) int64
    features: np.ndarray  # (N, T, 7) float32 strictly causal bin observations
    targets: np.ndarray  # (N, T) float32 normalized batch CPU demand
    raw_cpu_rate: np.ndarray  # (N, T) float32 unnormalized core-arrivals per bin
    raw_arr_count: np.ndarray  # (N, T) float32 task arrivals per bin
    bg_cpu_util: np.ndarray  # (N, T) float32 [0, 1] step-held server_usage CPU
    bg_mem_util: np.ndarray  # (N, T) float32 [0, 1] step-held server_usage Mem
    y_scales: np.ndarray  # (N,) float32 computed ONLY from train [0:train_end_bin]
    arr_scales: np.ndarray  # (N,) float32 computed ONLY from train [0:train_end_bin]
    machine_cpu_cap: np.ndarray  # (N,) float32 from server_event.csv
    task_events_path: Path  # Parquet file of real arrival instances for selected hosts
    feature_names: Tuple[str, ...] = (
        "norm_cpu_rate",
        "norm_arr_cnt",
        "bg_cpu_util",
        "bg_mem_util",
        "ema_short",
        "sin_tod",
        "cos_tod",
    )


def verify_schema_csv(schema_path: Path) -> Dict[str, List[str]]:
    """Verify official cluster-trace-v2017 schema.csv field layout."""
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing official schema.csv at {schema_path}")
    df = pd.read_csv(schema_path)
    df.columns = [c.strip() for c in df.columns]
    mapping: Dict[str, List[str]] = {}
    for fname, grp in df.groupby("file name"):
        ordered = grp.sort_values("field number")
        mapping[str(fname).strip()] = [str(x).strip() for x in ordered["content"].tolist()]
    for req_file, req_len in REQUIRED_SCHEMA_COLS.items():
        if req_file not in mapping or len(mapping[req_file]) != req_len:
            raise ValueError(
                f"Schema mismatch for {req_file}: expected {req_len} fields, got {len(mapping.get(req_file, []))}"
            )
    return mapping


def _greedy_diverse_machines(
    signal: np.ndarray, candidate_indices: np.ndarray, n_select: int
) -> np.ndarray:
    """
    Deterministic correlation-minimizing selection over qualified physical hosts
    so nodes represent heterogeneous, asynchronous real machines rather than IID clones.
    """
    if len(candidate_indices) < n_select:
        raise ValueError(
            f"Only {len(candidate_indices)} qualified hosts available, need {n_select}."
        )
    sub = signal[candidate_indices].astype(np.float64)
    sub = sub - sub.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-9
    sub = sub[valid] / norms[valid]
    cands = candidate_indices[valid]

    chosen_local = [0]
    max_corr = sub @ sub[0]
    for _ in range(1, n_select):
        score = max_corr.copy()
        for c in chosen_local:
            score[c] = 999.0
        nxt = int(np.argmin(score))
        chosen_local.append(nxt)
        max_corr = np.maximum(max_corr, sub @ sub[nxt])
    return cands[chosen_local]


def build_or_load_curated_trace(
    trace_cfg: TraceConfig, n_nodes: int, force_rebuild: bool = False
) -> CuratedTraceDataset:
    """
    Build (or load cached) strictly causal Phase 3 dataset for `n_nodes` physical Alibaba hosts.
    """
    verify_schema_csv(trace_cfg.schema_csv)

    proc_dir = trace_cfg.processed_dir
    proc_dir.mkdir(parents=True, exist_ok=True)
    npz_path = proc_dir / f"curated_{n_nodes}n_{trace_cfg.dt_bin_s}s.npz"
    tasks_path = proc_dir / f"tasks_{n_nodes}n.parquet"
    meta_path = proc_dir / f"metadata_{n_nodes}n.json"

    if not force_rebuild and npz_path.exists() and tasks_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("n_nodes") == n_nodes
            and meta.get("t_start_s") == trace_cfg.t_start_s
            and meta.get("t_end_s") == trace_cfg.t_end_s
            and meta.get("dt_bin_s") == trace_cfg.dt_bin_s
            and meta.get("lookback_L") == trace_cfg.lookback_L
        ):
            data = np.load(npz_path)
            return CuratedTraceDataset(
                n_nodes=n_nodes,
                n_bins=int(meta["n_bins"]),
                dt_bin_s=trace_cfg.dt_bin_s,
                t_start_s=trace_cfg.t_start_s,
                t_end_s=trace_cfg.t_end_s,
                train_end_bin=int(meta["train_end_bin"]),
                val_end_bin=int(meta["val_end_bin"]),
                lookback_L=trace_cfg.lookback_L,
                selected_machine_ids=data["selected_machine_ids"],
                features=data["features"],
                targets=data["targets"],
                raw_cpu_rate=data["raw_cpu_rate"],
                raw_arr_count=data["raw_arr_count"],
                bg_cpu_util=data["bg_cpu_util"],
                bg_mem_util=data["bg_mem_util"],
                y_scales=data["y_scales"],
                arr_scales=data["arr_scales"],
                machine_cpu_cap=data["machine_cpu_cap"],
                task_events_path=tasks_path,
            )

    # Step 1: Identify healthy physical hosts (full 144 server_usage rows, zero softerror)
    se = pd.read_csv(
        trace_cfg.raw_static_dir / "server_event.csv",
        header=None,
        names=["ts", "machine_id", "event_type", "event_detail", "cpu", "mem", "disk"],
    )
    bad_hosts = set(se.loc[se["event_type"] != "add", "machine_id"].astype(int).tolist())
    cap_map = (
        se.loc[se["event_type"] == "add"]
        .drop_duplicates("machine_id")
        .set_index("machine_id")["cpu"]
        .to_dict()
    )

    su = pd.read_csv(
        trace_cfg.raw_static_dir / "server_usage.csv",
        header=None,
        names=["ts", "machine_id", "cpu_util", "mem_util", "disk_util", "l1", "l5", "l15"],
    )
    su_counts = su.groupby("machine_id")["ts"].nunique()
    stable_machines = np.array(
        sorted(int(m) for m, c in su_counts.items() if c == 144 and int(m) not in bad_hosts),
        dtype=np.int64,
    )
    m_to_idx = {int(m): i for i, m in enumerate(stable_machines)}
    n_stable = len(stable_machines)

    t0, t1, dt = trace_cfg.t_start_s, trace_cfg.t_end_s, trace_cfg.dt_bin_s
    n_bins = (t1 - t0) // dt

    # Step 2: Stream batch_instance.csv once over [t0, t1) to qualify and diversify hosts
    arr_cnt_all = np.zeros((n_stable, n_bins), dtype=np.float32)
    cpu_rate_all = np.zeros((n_stable, n_bins), dtype=np.float32)

    for chunk in pd.read_csv(
        trace_cfg.batch_instance_csv,
        header=None,
        usecols=[0, 4, 9],  # start_ts, machine_id, average real cpu number
        chunksize=500_000,
        low_memory=False,
    ):
        s = pd.to_numeric(chunk[0], errors="coerce").to_numpy()
        m_raw = pd.to_numeric(chunk[4], errors="coerce").to_numpy()
        c_avg = pd.to_numeric(chunk[9], errors="coerce").fillna(0.5).to_numpy()
        ok = (~np.isnan(s)) & (~np.isnan(m_raw)) & (s >= t0) & (s < t1)
        if not ok.any():
            continue
        s = s[ok]
        m_ids = m_raw[ok].astype(np.int64)
        c_avg = np.clip(c_avg[ok], 0.05, 64.0).astype(np.float32)

        idx = np.array([m_to_idx.get(int(x), -1) for x in m_ids], dtype=np.int32)
        keep = idx >= 0
        if not keep.any():
            continue
        idx = idx[keep]
        b = ((s[keep] - t0) // dt).astype(np.int32)
        np.add.at(arr_cnt_all, (idx, b), 1.0)
        np.add.at(cpu_rate_all, (idx, b), c_avg[keep])

    # Qualify by 60s-equivalent arrival density cleanly (>=80% nonzero minutes, >=6/min)
    # Note: pair adjacent bins if dt == 30 so threshold mirrors Stage 1 audit directly.
    bins_per_min = max(1, 60 // dt)
    usable_cols = (n_bins // bins_per_min) * bins_per_min
    arr_min = arr_cnt_all[:, :usable_cols].reshape(n_stable, -1, bins_per_min).sum(axis=2)
    nz_min = (arr_min > 0).mean(axis=1)
    mean_min = arr_min.mean(axis=1)

    cands = np.where((nz_min >= trace_cfg.min_nonzero_frac) & (mean_min >= trace_cfg.min_mean_tasks_per_min))[0]
    cands = cands[np.argsort(mean_min[cands])[::-1]]
    pool = cands[: min(trace_cfg.candidate_pool_limit, len(cands))]
    chosen_local = _greedy_diverse_machines(cpu_rate_all, pool, n_nodes)
    selected_machines = stable_machines[chosen_local]
    sel_map = {int(m): i for i, m in enumerate(selected_machines)}

    raw_cpu_rate = cpu_rate_all[chosen_local].copy()
    raw_arr_count = arr_cnt_all[chosen_local].copy()
    machine_cpu_cap = np.array([float(cap_map.get(int(m), 64.0)) for m in selected_machines], dtype=np.float32)

    # Step 3: Extract real per-task arrival records for ONLY selected_machines -> Parquet
    task_chunks = []
    for chunk in pd.read_csv(
        trace_cfg.batch_instance_csv,
        header=None,
        usecols=[0, 1, 2, 3, 4, 5, 8, 9, 10],
        chunksize=500_000,
        low_memory=False,
    ):
        s = pd.to_numeric(chunk[0], errors="coerce")
        m_raw = pd.to_numeric(chunk[4], errors="coerce")
        mask = s.notna() & m_raw.notna() & (s >= t0) & (s < t1) & m_raw.isin(sel_map)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub.columns = [
            "start_ts",
            "end_ts",
            "job_id",
            "task_id",
            "machine_id",
            "status",
            "max_real_cpu",
            "avg_real_cpu",
            "max_norm_mem",
        ]
        sub["start_ts"] = pd.to_numeric(sub["start_ts"], errors="coerce").astype(np.float64)
        sub["end_ts"] = pd.to_numeric(sub["end_ts"], errors="coerce").fillna(sub["start_ts"]).astype(np.float64)
        sub["duration_s"] = np.clip(sub["end_ts"] - sub["start_ts"], 1.0, 3600.0).astype(np.float32)
        sub["avg_real_cpu"] = pd.to_numeric(sub["avg_real_cpu"], errors="coerce").fillna(0.5).clip(0.05, 64.0).astype(np.float32)
        sub["max_norm_mem"] = pd.to_numeric(sub["max_norm_mem"], errors="coerce").fillna(0.015).clip(0.001, 0.5).astype(np.float32)
        sub["cpu_work"] = (sub["avg_real_cpu"] * sub["duration_s"]).astype(np.float32)
        sub["machine_id"] = sub["machine_id"].astype(np.int64)
        sub["node_idx"] = sub["machine_id"].map(sel_map).astype(np.int32)
        sub["job_id"] = pd.to_numeric(sub["job_id"], errors="coerce").fillna(-1).astype(np.int64)
        sub["task_id"] = pd.to_numeric(sub["task_id"], errors="coerce").fillna(-1).astype(np.int64)
        sub["status"] = sub["status"].astype(str)
        task_chunks.append(
            sub[
                [
                    "start_ts",
                    "duration_s",
                    "cpu_work",
                    "avg_real_cpu",
                    "max_norm_mem",
                    "node_idx",
                    "machine_id",
                    "job_id",
                    "task_id",
                    "status",
                ]
            ]
        )

    tasks_df = pd.concat(task_chunks, ignore_index=True).sort_values(["start_ts", "node_idx"]).reset_index(drop=True)
    tasks_df.to_parquet(tasks_path, index=False)

    # Step 4: Causal backward-held server_usage (ts <= bin close t_close)
    su_sel = su[su["machine_id"].isin(selected_machines)].copy()
    su_ts = np.sort(su_sel["ts"].unique()).astype(np.float64)
    su_cpu_piv = (
        su_sel.pivot(index="machine_id", columns="ts", values="cpu_util")
        .reindex(index=selected_machines, columns=su_ts)
        .ffill(axis=1)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        / 100.0
    )
    su_mem_piv = (
        su_sel.pivot(index="machine_id", columns="ts", values="mem_util")
        .reindex(index=selected_machines, columns=su_ts)
        .ffill(axis=1)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        / 100.0
    )
    bin_close_s = t0 + (np.arange(n_bins, dtype=np.float64) + 1.0) * dt
    causal_su_idx = np.clip(np.searchsorted(su_ts, bin_close_s, side="right") - 1, 0, len(su_ts) - 1)
    bg_cpu_util = su_cpu_piv[:, causal_su_idx].astype(np.float32)
    bg_mem_util = su_mem_piv[:, causal_su_idx].astype(np.float32)

    # Step 5: Train-only normalization & strictly causal feature matrix
    train_end_bin = int(trace_cfg.train_frac * n_bins)
    val_end_bin = int((trace_cfg.train_frac + trace_cfg.val_frac) * n_bins)

    y_scales = np.zeros(n_nodes, dtype=np.float32)
    arr_scales = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        ys = float(np.percentile(raw_cpu_rate[i, :train_end_bin], 99))
        y_scales[i] = max(1e-3, ys)
        ars = float(np.percentile(raw_arr_count[i, :train_end_bin], 99))
        arr_scales[i] = max(1e-3, ars)

    norm_cpu = np.clip(raw_cpu_rate / y_scales[:, None], 0.0, 1.5).astype(np.float32)
    norm_arr = np.clip(raw_arr_count / arr_scales[:, None], 0.0, 1.5).astype(np.float32)

    # Causal short EMA (alpha = 0.35)
    ema = np.zeros_like(norm_cpu)
    ema[:, 0] = norm_cpu[:, 0]
    alpha = 0.35
    for t in range(1, n_bins):
        ema[:, t] = alpha * norm_cpu[:, t] + (1.0 - alpha) * ema[:, t - 1]

    tod_rad = 2.0 * np.pi * ((bin_close_s % 86400.0) / 86400.0)
    sin_tod = np.broadcast_to(np.sin(tod_rad).astype(np.float32), (n_nodes, n_bins))
    cos_tod = np.broadcast_to(np.cos(tod_rad).astype(np.float32), (n_nodes, n_bins))

    features = np.stack(
        [norm_cpu, norm_arr, bg_cpu_util, bg_mem_util, ema, sin_tod, cos_tod],
        axis=-1,
    ).astype(np.float32)
    targets = norm_cpu.astype(np.float32)

    np.savez_compressed(
        npz_path,
        selected_machine_ids=selected_machines,
        features=features,
        targets=targets,
        raw_cpu_rate=raw_cpu_rate,
        raw_arr_count=raw_arr_count,
        bg_cpu_util=bg_cpu_util,
        bg_mem_util=bg_mem_util,
        y_scales=y_scales,
        arr_scales=arr_scales,
        machine_cpu_cap=machine_cpu_cap,
    )

    meta = {
        "n_nodes": n_nodes,
        "n_bins": n_bins,
        "dt_bin_s": dt,
        "t_start_s": t0,
        "t_end_s": t1,
        "train_end_bin": train_end_bin,
        "val_end_bin": val_end_bin,
        "lookback_L": trace_cfg.lookback_L,
        "total_task_events": int(len(tasks_df)),
        "selected_machine_ids": selected_machines.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CuratedTraceDataset(
        n_nodes=n_nodes,
        n_bins=n_bins,
        dt_bin_s=dt,
        t_start_s=t0,
        t_end_s=t1,
        train_end_bin=train_end_bin,
        val_end_bin=val_end_bin,
        lookback_L=trace_cfg.lookback_L,
        selected_machine_ids=selected_machines,
        features=features,
        targets=targets,
        raw_cpu_rate=raw_cpu_rate,
        raw_arr_count=raw_arr_count,
        bg_cpu_util=bg_cpu_util,
        bg_mem_util=bg_mem_util,
        y_scales=y_scales,
        arr_scales=arr_scales,
        machine_cpu_cap=machine_cpu_cap,
        task_events_path=tasks_path,
    )


def make_supervised_windows(
    features: np.ndarray,
    targets: np.ndarray,
    lookback_L: int,
    start_bin: int,
    end_bin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build strictly causal sliding windows for ONE node:
    Input X[i] = features[start..end-1][t - L : t, :] -> Target y[i] = targets[t]
    where t runs over [max(lookback_L, start_bin), end_bin).
    Zero future bin overlap between X[t] and y[t]!
    """
    t0 = max(lookback_L, start_bin)
    xs = []
    ys = []
    for t in range(t0, end_bin):
        xs.append(features[t - lookback_L : t])
        ys.append(targets[t])
    if not xs:
        return np.empty((0, lookback_L, features.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.float32)
