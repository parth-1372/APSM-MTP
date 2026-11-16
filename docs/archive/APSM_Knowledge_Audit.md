# Complete APSM Project Knowledge Audit

> Audit based strictly on direct inspection of source code, configurations, result CSVs/JSONs, execution logs, PNG plots, and documentation in `d:/coolage study materials/MTP`.

---

## 1. APSM Architecture Reconstruction

### 1.1 Problem Solved by the Original Tundo et al. (2025) Gossip Learning Approach
- **Target problem:** Fully decentralized, serverless time-series workload forecasting across edge nodes without a central aggregator (avoiding Federated Learning's star-topology single point of failure and server bottleneck).
- **In our repository:** Nodes sit on an undirected $k$-edge-augmented peer-to-peer (P2P) graph and train local LSTM regressors without sharing raw time-series data, exchanging only model parameters with graph neighbors.

### 1.2 How Our Existing Gossip Learning Implementation Works
- Implemented as a **single-process discrete-event simulation (DES)** inside [`Simulator`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L36-L205), backed by Python's `queue.PriorityQueue` ordered chronologically by discrete integer simulation seconds (`Event.time`).
- At initialization ([`Simulator.__init__`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L41-L91)), it creates `N` [`Node`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L37-L414) instances, each holding:
  - Its own Keras LSTM instance `self._model` initialized with fresh random weights (`self._best_val_loss = inf`, `self.accumulated_weight = 0`).
  - Its local partitioned dataset `self.data` via [`get_node_dataset`](file:///d:/coolage%20study%20materials/MTP/src/utils/gossip_training.py#L26-L43).
  - Outgoing neighbor links [`Link`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/models.py#L63-L70) (`weights_transmission_time=30`, `round_trip_time=1`).
- At simulation start ([`Simulator.run_training_simulation`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L93-L190)), each node `i` schedules an initial [`SendModelsLoopEvent`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L35-L37) at a random staggered start time `t ∈ {0, ..., 59}` ([`simulator.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L102-L104)).
- **Important bootstrap behavior:** Nodes **do not train before their first send**. They broadcast untrained initial weights (`model_weight = 0`) until receiving `num_merged_models` (default `1`) peer messages triggers local merging and training.

### 1.3 How Nodes Train and Exchange LSTM Models
1. **Transmission ([`process_send_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L145-L238)):**
   - Filters `node.active_links` to remove peers already in `history.stopped_time`.
   - Samples `ceil(len(active_links) * target_probability)` neighbors without replacement (in [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L138), `target_probability = 1.0`, i.e., all active neighbors).
   - Calls [`Node.marshal_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L271-L285), which calls [`marshal_weights_with_random_subsampling`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/weights_marshaling.py#L26-L46) (`perc_sent_weights = 1.0` sends 100% of flattened parameters; `serialize_optimizer = False`).
   - Enqueues one [`ReceiveModelEvent`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L40-L45) per selected neighbor at `t + weights_transmission_time` (`t + 30`), logs each in `history.messages` as a [`MessageHistoryLog`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/history.py#L9-L14), and reschedules [`SendModelsLoopEvent`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L35-L37) for itself at `t + max(weights_transmission_time)` (`t + 30`).
2. **Reception & Triggering Training ([`process_receive_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L241-L306)):**
   - Discards incoming weights if `node.state == FAILED`, `node.id ∈ history.stopped_time`, or `node.state == TRAINING` (`discarded` while node is in its 5s training window).
   - Otherwise buffers the message via [`Node.receive_weights`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L372-L379).
   - When `node.ready_to_train` (`len(self._received_weights) >= num_merged_models`) holds:
     - Calls [`Node.merge_models`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L116-L129).
     - Calls [`Node.perform_update`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L191-L217), setting `self.state = TRAINING` and running [`Node.train_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L219-L269) for `epochs_per_update = 3`.
     - Inside `train_model`, **a fresh Keras model `model = self._create_model()` is instantiated on every update**, initialized with `self._model.get_weights()`, and trained via 3 separate `epochs=1` calls to `model.fit(...)` (which resets Adam's moment accumulators every update and records `best_weights` / `best_val_loss` across the 3 epochs alongside `latest_weights`).
     - Immediately increments `self._completed_updates` and calls [`Node._evaluate`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L381-L390) on `self._model`.
     - Schedules [`SaveModelEvent`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L47-L55) at `t + TRAIN_TIME` (`t + 5`), carrying `latest_weights`, `best_update_model_weights`, `best_update_val_loss`, and `new_weight = node.accumulated_weight + node.weight`.
3. **Saving ([`process_save_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L309-L359) & [`Node.save_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L287-L350)):**
   - First runs APSM's [`Node.update_semantic_state`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L134-L154) **before** mutating `self._best_val_loss`.
   - Overwrites `self._model` with `latest_weights`, updates `self.accumulated_weight = new_model_weight`, and updates `self._best_weights` / `self._best_val_loss` only if `best_update_val_loss < self._best_val_loss`.
   - Checks [`Node._check_stop_criterion`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L399-L414); once `completed_updates >= fixed_updates` (100), sets `state = STOPPED` and records `history.stopped_time[node.id] = event.time`.
   - After the DES loop exits, [`Simulator.run_training_simulation`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L166-L176) runs `finetuning_epochs = 2` (default from [`config.json`](file:///d:/coolage%20study%20materials/MTP/src/config.json#L17)) locally per node, updates `_best_weights`, and saves `{node.id}.h5` via [`Node.persist_best_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L357-L370) — **without** calling `_evaluate()`.

### 1.4 Model Aggregation and Gossip Communication
- Selected via [`choose_aggregator`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L204-L221). In [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L136), `merge_strategy = MergeStrategy.AGE_WEIGHTED` uses [`merge_weights_with_age_weighted_avg`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L56-L101):
  - For each peer message `msg`: `w_peer = msg.model_weight if msg.model_weight > 0 else 1`.
  - For the local node (`node_model_age`): adds `node_model_age` directly to `ages_sum` ([`aggregators.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L87)), while multiplying weights by `(node_model_age if node_model_age > 0 else 1)` ([`aggregators.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L88)). Then `ages_sum[ages_sum == 0] = 1`, followed by `sums / ages_sum`.
  - Sets `new_model_age = max(node_model_age, *[mw.model_weight for mw in weights_messages])`.
  - Note: `node.weight` via [`weight_by_dataset_size`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/weight.py#L6-L7) is `len(X_train)` (`21,946` per node), so `accumulated_weight` increments in multiples of `21,946`.

### 1.5 What APSM Changes Compared with Standard Gossip Learning
APSM adds pre-send gating state in [`Node`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L101-L189), config fields (`is_baseline`, `semantic_k`, `semantic_window`, `semantic_heartbeat`) in [`TrainingConfig`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/config.py#L149-L185), two telemetry dictionaries (`suppressed_packets`, `semantic_surprise_scores`) in [`History`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/history.py#L40-L44), and two hooks in [`event.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py):
1. Pre-send gate in [`process_send_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L159-L173) calling [`Node.should_suppress_transmission`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L156-L189).
2. Post-training semantic update in [`process_save_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L332-L343) calling [`Node.update_semantic_state`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L134-L154).
When `is_baseline=True`, [`should_suppress_transmission`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L170-L172) always returns `False` (reducing to standard GL). **Nothing in training, LSTM structure, or aggregation changes.**

### 1.6 Exact Semantic-Surprise Calculation ([`Node.update_semantic_state`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L134-L154))
Given `best_update_val_loss` (the minimum validation MSE across the `3` local epochs in [`Node.train_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L250-L257)), computed **before** [`Node.save_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L344-L351) updates `self._best_val_loss`:
$$\varepsilon(t) = \begin{cases} \left|\text{best\_update\_val\_loss} - \text{self.\_best\_val\_loss}\right|, & \text{if } \text{self.\_best\_val\_loss} < \infty \\ \text{best\_update\_val\_loss}, & \text{if } \text{self.\_best\_val\_loss} = \infty \text{ (update 1)} \end{cases}$$
- **Critical verification:** Surprise is the **absolute delta in scalar validation MSE relative to the node's lifetime best validation MSE** — **not** test-time prediction residual $|y_t - \hat{y}_t|$ as claimed in [`presentation_script.md`](file:///d:/coolage%20study%20materials/MTP/presentation_script.md#L66).

### 1.7 Adaptive Threshold Calculation ([`Node.update_semantic_state`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L147-L154))
- Appends $\varepsilon(t)$ to `self._error_window` (`collections.deque(maxlen=semantic_window)`, where `semantic_window = 50`).
- Computes:
$$\tau(t) = \begin{cases} k \cdot \text{np.std}(\text{\_error\_window}), & \text{if } |\text{\_error\_window}| \ge 2 \text{ (population std, } \text{ddof}=0\text{)} \\ \infty, & \text{if } |\text{\_error\_window}| < 2 \end{cases}$$
with `k = 2.0`.
- **Critical verification:** Filtering engages at **update 2** (`len >= 2`), contradicting [`experiment_config.py`](file:///d:/coolage%20study%20materials/MTP/src/experiment_config.py#L44) and [`RESEARCH_REPORT.md`](file:///d:/coolage%20study%20materials/MTP/docs/RESEARCH_REPORT.md#L593) which imply 50 updates are needed.

### 1.8 Heartbeat Mechanism and Its Purpose ([`Node.should_suppress_transmission`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L178-L189))
- Computes `suppress = (self.last_surprise <= self.last_threshold)`.
- On `suppress == True`, increments `self.consecutive_suppressions += 1`; otherwise resets `self.consecutive_suppressions = 0`.
- If `self.consecutive_suppressions >= self._semantic_heartbeat` (`5` in [`experiment_config.py`](file:///d:/coolage%20study%20materials/MTP/src/experiment_config.py#L67); default `3` in [`TrainingConfig`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/config.py#L178)), resets counter to `0` and returns `False` (forcing a broadcast).
- **Purpose:** Prevents network deadlock/starvation in a gossip topology where nodes only train when receiving peer models (`ready_to_train`); if every node suppressed indefinitely, no new `ReceiveModelEvent` would arrive and no node could ever reach `fixed_updates = 100`.

### 1.9 Which Model Weights Are Actually Transmitted
- [`Node.marshal_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L277-L285) serializes `self._model`.
- [`Node.save_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L315) executes `self._model.set_weights(latest_weights)`.
- Therefore, **gossip transmissions always carry `latest_weights` (epoch 3 of the latest local update)**, **not** `_best_weights` (which is stored separately in `self._best_weights` and written only to `{id}.h5` at the very end via [`persist_best_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L357-L370)).
- **Notable decoupling:** Semantic surprise uses `best_update_val_loss` (epoch-best across the 3 epochs), while the transmitted payload is `latest_weights` (epoch 3).

### 1.10 Simulation Events and Network Communication Handling
- Six dataclasses in [`event.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L15-L55) dispatched via `@singledispatch` [`process_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L60-L69):
  - `SendModelsLoopEvent`: Checks `FAILED` (not `STOPPED`!), runs APSM gate (if suppressed: increments `suppressed_packets[node.id]` by 1 and reschedules self at `t + 5s`; if sent: schedules `ReceiveModelEvent` for all selected peers at `t + 30s`, logs in `history.messages`, reschedules self at `t + 30s`).
  - `ReceiveModelEvent`: Link failure drop hook in [`Simulator.run_training_simulation`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L120-L122) (if `LINK_FAILURE`); ignores if `FAILED`, stopped, or `TRAINING`; buffers weights; when ready, merges, runs synchronous `perform_update()`, schedules `SaveModelEvent` at `t + 5s`.
  - `SaveModelEvent`: Ignores if `FAILED` or stopped; updates APSM state, records `(time, surprise, threshold)` in `history.semantic_surprise_scores`, saves weights, checks stopping criterion.
  - `IsTimeToFailEvent`, `FailedNodeEvent`, `RecoveryNodeEvent`: Present for fault injection (`FailureMode.NODE_FAILURE`), disabled when `failure_mode = "none"` (default in [`TrainingConfig`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/config.py#L77-L79)).

---

## 2. Experimental Setup Reconstruction

### 2.1 Dataset Generation and Preprocessing
- **Network cell coordinates only (`assets/porto_cells.parquet`):** Cleaned OpenCellID antenna coordinates inside bounding box `(-8.6338, -8.5862, 41.1369, 41.1690)` from [`assets/BBox_Porto.txt`](file:///d:/coolage%20study%20materials/MTP/assets/BBox_Porto.txt), built in [`0_cell_data_preparation.ipynb`](file:///d:/coolage%20study%20materials/MTP/src/0_cell_data_preparation.ipynb#L457-L601).
- **Time-series workloads are purely synthetic ([`2_dataset_generation.py`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py)) — NOT Porto Taxi GPS trajectories or Microsoft Azure Serverless traces as claimed in docs:**
  1. Generates a 15-minute `pd.date_range` from `2018-01-01` to `2019-01-01` (`35,041` steps).
  2. Constructs an "ideal" diurnal trend per tower using [`Akima1DInterpolator`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L86-L156) over 5 weekly anchor times (`base_fn = 500`, scaled linearly by normalized Euclidean distance from Porto BBox center via [`get_traffic_function`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L162-L178)).
  3. Multiplies by shared multivariate AR(4) noise (`0.05 * time_series * ideal_trend + ideal_trend`) and drops first 200 samples -> `34,841` rows ([`get_tower_real_trends`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L309-L334)).
  4. Clips each node at `[0, 95th percentile]` ([`clip_datasets`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L342-L346)) and globally scales by maximum across all nodes into `[0, 1]` ([`scale_datasets`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L354-L363)). No `scaling_factor.txt` or `305` multiplier exists in `2_dataset_generation.py`.
  5. Encodes `hour` and `weekday` with `category_encoders.BinaryEncoder` into 8 binary features ([`encode_time`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L371-L374)), yielding **9 features per timestep**.
  6. Creates `(4 -> 1)` sliding windows (`34,837` rows) via [`encode_sequences_for_training`](file:///d:/coolage%20study%20materials/MTP/src/utils/data.py#L11-L34) and chronologically splits with `test_perc=0.3`, `val_perc_on_train=0.1` in [`train_val_test_split`](file:///d:/coolage%20study%20materials/MTP/src/utils/data.py#L37-L49): **`21,946` train (63%) / `2,439` val (7%) / `10,452` test (30%)** per node (confirmed in [`phase2_run.log`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full/phase2_run.log#L2-L11)).

### 2.2 Number of Nodes and Network Topology
- [`1_network_generation.py`](file:///d:/coolage%20study%20materials/MTP/src/1_network_generation.py#L29-L76) samples `N_NODES ∈ {10, 50, 100}` towers without replacement from `porto_cells.parquet`, computes pairwise Haversine distances, and applies `nx.k_edge_augmentation(network, k=3, avail=distances)` to construct an undirected **3-edge-connected sparse graph** (`data/networks/porto_10n_3k/0/adj_list.txt` verified on disk; `50n_3k` and `100n_3k` topologies/datasets are absent locally and were likely generated on Kaggle/elsewhere).
- Note [`adj_list.txt`](file:///d:/coolage%20study%20materials/MTP/data/networks/porto_10n_3k/0/adj_list.txt) is undirected ($k$-edge-augmented, not strictly $k$-regular `degree=3`: e.g., Node 0 has degree 4 `[7, 8, 4, 5]`, Node 1 has degree 3 `[9, 3, 2]`).

### 2.3 LSTM Architecture and Training Parameters
- Defined in [`create_LSTM`](file:///d:/coolage%20study%20materials/MTP/src/utils/model_creators.py#L10-L48):
  - `Input(shape=(4, 9))` -> `LSTM(50, tanh, return_sequences=True)` -> `LSTM(50, tanh, return_sequences=False)` -> `Dropout(0.2)` -> `Dense(32, relu)` -> `Dropout(0.2)` -> `Dense(1, relu, name="fn_0", dtype="float32")`.
  - Optimizer: `Adam(learning_rate=0.001, epsilon=1e-6)`, Loss: `MSE`.
- Training parameters in [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L126-L151) + [`experiment_config.py`](file:///d:/coolage%20study%20materials/MTP/src/experiment_config.py) + [`config.json`](file:///d:/coolage%20study%20materials/MTP/src/config.json):
  - `fixed_updates = 100`, `epochs_per_update = 3`, `batch_size = 128`, `shuffle_batch = True`, `merge_strategy = "age_weighted"`, `target_probability = 1.0`, `perc_sent_weights = 1.0`, `num_merged_models = 1`, `finetuning_epochs = 2` (from `config.json`), `failure_mode = "none"`, `eval_test = True`, `freq = 5`.
  - APSM parameters: `semantic_k = 2.0`, `semantic_window = 50`, `semantic_heartbeat = 5`.

### 2.4 Number of Seeds and Experiment Configurations
- **Canonical 3-way scalability runs (`10n`, `50n`, `100n`):** `N_SEEDS = 1` (`seed = 0` only, topology `simulation_number = 0`), `MAX_SAMPLES = None` (full `21,946` train samples/node), `N_SMALL_DATASET_NODES = 0`, `RUN_FEDERATED = True`, `FL_ROUNDS = 100`, `FL_EPOCHS = 3`, `FL_FRACTION_FIT = 1.0`.
- **Legacy/Unused configs & scripts:**
  - [`config.json`](file:///d:/coolage%20study%20materials/MTP/src/config.json) defaults (`epochs=10, batch=64, simple_avg, fixed_updates=30, target_prob=0.5`) — mostly overridden by `run_all_comparisons.py` (except `finetuning_epochs=2`, `perc_sent_weights=1`, `num_merged_models=1`, `serialize_optimizer=False`, `failure_mode=none`).
  - [`configs/scalability_simulation_config.json`](file:///d:/coolage%20study%20materials/MTP/src/configs/scalability_simulation_config.json) (`50n, fixed_updates=25, age_weighted`, used by [`scalability_simulation.py`](file:///d:/coolage%20study%20materials/MTP/src/scalability_simulation.py) which fails because it looks for `scalability_simulation_config.json` in `src/` root).
  - [`threshold_sweep.py`](file:///d:/coolage%20study%20materials/MTP/src/threshold_sweep.py) (`K_VALUES = [0.5..5.0]`) is **broken/unrun** (patches non-existent `self._SEMANTIC_K` instead of `_semantic_k`, points to missing `seed1000/0` and missing `experiments/gl_baseline/0/history.json`).

### 2.5 Baseline Methods
1. **GL-Baseline (Tundo 2025):** Same [`Simulator`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py) with `is_baseline=True`.
2. **Federated Learning (FedAvg):** Implemented in a custom TensorFlow loop [`run_federated_simulation`](file:///d:/coolage%20study%20materials/MTP/src/utils/federated_training.py#L30-L210) (star topology with server `-1`, `fraction_fit=1.0`, `n_rounds=100`, `epochs=3`, `batch_size=128`). **Crucial finding:** The separate Flower package [`fl-baseline/`](file:///d:/coolage%20study%20materials/MTP/fl-baseline/fl_baseline/server_app.py) is **not used** by `run_all_comparisons.py`.
3. **Other Baselines in Repo (Unused in Phase 2 Benchmarks):** Centralized training ([`train_centralized.py`](file:///d:/coolage%20study%20materials/MTP/src/train_centralized.py)) and standalone per-node training ([`train_single.py`](file:///d:/coolage%20study%20materials/MTP/src/train_single.py)).

### 2.6 Evaluation Metrics & Methodological Asymmetry
- Computed via [`compute_metrics`](file:///d:/coolage%20study%20materials/MTP/src/utils/metrics.py#L40-L55) (GL/APSM) and NumPy formulas in [`federated_training.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/federated_training.py#L176-L197) (FL): MSE, RMSE (`mean` of per-node final RMSEs — not $\sqrt{\text{mean MSE}}$ across heterogeneous nodes), and MAE.
- [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L334) multiplies scaled MSE by `SCALE = 305 ** 2 = 93025` to report "raw, m²" MSE, even though `305` appears nowhere in the synthetic data generator.
- **Severe FL vs. GL/APSM Evaluation Asymmetry:**
  - `GL-Baseline` and `APSM` pass `test_set = common_test_set` (10% pooled `X_test/Y_test` across all `N` nodes) into `Node`, so `history["nodes_test_history"]` evaluates each node's **pre-training merged model** on the **shared common test set**.
  - [`run_federated_simulation`](file:///d:/coolage%20study%20materials/MTP/src/utils/federated_training.py#L173-L198) writes the `common_test_set` evaluation into `global_test_history`, while writing the global model's evaluation on each node's **local** `X_test/Y_test` into `nodes_test_history` (after round-end FedAvg aggregation, with persistent client Adam optimizers).
  - [`extract_metrics`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L263-L295) reads `nodes_test_history` **first**! Thus **reported FL MSEs are post-aggregation local-test-set averages, while GL/APSM MSEs are pre-local-training common-test-set averages.** Visual inspection of `graph2_convergence_3way.png` confirms multiple FL nodes' MSE curves monotonically **climb** across rounds on their non-IID local test sets (`>0.07` on 10n, `>0.03` on 100n), proving local-set drift rather than global convergence.

### 2.7 Packet-Counting Methodology (Critical Timing & Unit Mismatch)
- **GL-Baseline / APSM `total_sent`:** `len(history["messages"])`, counting individual node-to-neighbor [`MessageHistoryLog`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/history.py#L9-L14) edges (~3 per broadcast decision).
- **FL `total_sent`:** `2 * N_NODES * FL_ROUNDS` (`1` downlink + `1` uplink log per client per round).
- **APSM `suppressed`:** Incremented by `+1` in [`process_send_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L162-L167) per **suppressed broadcast decision** (not multiplied by neighbor degree $k$), AND suppression reschedules the next poll at `t + 5s` (`TRAIN_TIME`) instead of `t + 30s` (`weights_transmission_time`) — **polling 6× more frequently whenever suppressed!**
- **"Packet Reduction vs GL" formula ([`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L327-L328), [L602-L603](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L602-L603)):**
  $$\text{Reported Packet Reduction (\%)} = \frac{\text{APSM\_suppressed\_decisions}}{\text{GL\_baseline\_messages\_sent} + \text{APSM\_suppressed\_decisions}} \times 100$$
  This mixes GL-Baseline's neighbor-multiplied physical packets with APSM's 5-second decision-suppression counter (even when APSM simulated longer, e.g., `2540s` vs `2035s` in `phase2_run.log`). It does **not** measure actual physical packets saved `(GL_sent - APSM_sent) / GL_sent`.

### 2.8 Forecasting Error Measurements & Scalability Experiments
- `nodes_test_history` records MSE, RMSE, MSLE, MAE every `freq = 5` updates (20 evaluation points across 100 updates). Note that [`extract_metrics`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L271) takes only the final point `[-1]`.
- Scalability runs execute [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py) across `N_NODES ∈ {10, 50, 100}` (`K = 3`, `N_SEEDS = 1`, `seed = 0`).

---

## 3. Actual Experimental Results (Strictly Separated by Provenance)

Three mutually distinct sets of numbers exist across the repository. They must **never** be blended.

### 3.1 Run Set 1 — Verified 3-Way Scalability Runs (`results/phase2_full_{10n,50n,100n}/`)
Artifacts: [`three_way_comparison_full.csv`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/three_way_comparison_full.csv) & [`three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/three_way_summary.json) (10n, `2026-10-18`); [50n CSV](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/three_way_comparison_full.csv) & [50n JSON](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/three_way_summary.json) (`2026-10-19`); [100n CSV](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/three_way_comparison_full.csv) & [100n JSON](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/three_way_summary.json) (`2026-10-24`). Config across all three: `seed=0` (`N_SEEDS=1`), `fixed_updates=100`, `epochs=3`, `k=2.0`, `window=50`, `heartbeat=5`.

| Scale | Method | Avg MSE (scaled) | Avg RMSE (scaled) | Avg MAE (scaled) | Total Pkts Sent | Suppressed Decisions | Reported Reduction % | **Actual Pkts Saved vs GL (%)** | MSE Δ vs GL (%) |
|---|---|---|---|---|---|---|---|---|---|
| **10n** | FL (FedAvg)* | `0.021337` | `0.116352` | `0.103060` | `2,000` | `0` | — | — | — |
| **10n** | GL-Baseline | `0.031063` | `0.173967` | `0.121294` | `1,437` | `0` | `0.0%` | `0.0%` | `0.0%` |
| **10n** | APSM Phase 2 | `0.032376` | `0.177579` | `0.122437` | `1,305` | `1,702` | `54.22%` | **`9.19%`** (`132` pkts) | **`+4.23%`** |
| **50n** | FL (FedAvg)* | `0.005617` | `0.063647` | `0.056632` | `10,000` | `0` | — | — | — |
| **50n** | GL-Baseline | `0.010750` | `0.102172` | `0.081077` | `8,029` | `0` | `0.0%` | `0.0%` | `0.0%` |
| **50n** | APSM Phase 2 | `0.011083` | `0.103373` | `0.082029` | `6,652` | `8,112` | `50.26%` | **`17.15%`** (`1,377` pkts) | **`+3.10%`** |
| **100n** | FL (FedAvg)* | `0.005880` | `0.067639` | `0.058186` | `20,000` | `0` | — | — | — |
| **100n** | GL-Baseline | `0.007967` | `0.087914` | `0.066889` | `15,421` | `0` | `0.0%` | `0.0%` | `0.0%` |
| **100n** | APSM Phase 2 | `0.008495` | `0.091549` | `0.069661` | `13,240` | `15,769` | `50.56%` | **`14.14%`** (`2,181` pkts) | **`+6.64%`** *(fails `<=5%`)* |

*\*FL evaluated on per-node local test sets post-aggregation, unlike GL/APSM (common test set pre-training).*

### 3.2 Run Set 2 — Verified 2-Way Kaggle 10-Node Console Log (`results/phase2_full/phase2_run.log`)
Artifact: [`phase2_run.log`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full/phase2_run.log) (10 nodes, `seed=0`, `updates=100`, `epochs=3`, `k=2.0`, `window=50`, `heartbeat=5`, run path `/kaggle/working/APSM-MTP/results/phase2_full`, `125.1 min` total). Its output CSV/JSON/plots do **not** exist in the repo.
- **GL-Baseline** (ended `sim_t=2035s`, `events=3,011`): MSE `0.00582` (`542 m²`), RMSE `0.07571`, `1,447` packets sent.
- **APSM** (ended `sim_t=2540s`, `events=4,205`): MSE `0.00545` (`507 m²`), RMSE `0.07352`, `1,254` packets sent, `1,491` suppressed decisions.
- **MSE delta:** **`-6.34%`** (APSM *improved* MSE vs GL), yet console printed `❌ MSE degradation <= 5% -> 6.34%` due to `abs(mse_delta_vs_gl) <= 5`.
- **Packet reduction:** Reported `50.7%`; actual sent-packet reduction vs GL = **`13.34%`** (`193 / 1447`).

### 3.3 Run Set 3 — Unreconciled / Unverified Claims in Docs (No Backing Code/Data/CSVs in Repo)
1. **Docs' "June 30, 2026 7-Hour Kaggle Run"** ([`README.md`](file:///d:/coolage%20study%20materials/MTP/README.md#L24-L33), [`RESEARCH_REPORT.md`](file:///d:/coolage%20study%20materials/MTP/docs/RESEARCH_REPORT.md#L547-L565)): Claims GL MSE `0.009269` (`1,447` pkts) vs APSM `0.009332` (`+0.69%`), `1,314` sent, `1,674` suppressed (`53.6%`), and FL `~0.00918`. Referenced `src/experiments_full/` and `results/Phase2_Full/{phase2_comparison_full.csv, summary.json}` **do not exist**. Furthermore, [`RESEARCH_REPORT.md`](file:///d:/coolage%20study%20materials/MTP/docs/RESEARCH_REPORT.md#L581) falsely attributes the difference vs `phase2_run.log` to `finetuning_epochs=2`, whereas fine-tuning never triggers `_evaluate()`.
2. **MTP-1 Presentation Script** ([`presentation_script.md`](file:///d:/coolage%20study%20materials/MTP/presentation_script.md#L55-L83)): Claims `3,204` vs `1,614` (`49.61%`), `0.0318` vs `0.0332` (`+4.50%`), **`100,200 J` vs `50,490 J` energy**, 1-layer 16-unit LSTM, Microsoft Azure + Porto dataset, and prediction-error trigger. **Zero energy computation code or supporting logs exist in the repository** (`2022IMT082-MTP1(1).pdf` is also absent locally).

### 3.4 Visual & Artifact Discrepancies Verified in Existing Plots
- [`graph4_semantic_surprise.png` (10n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/graph4_semantic_surprise.png), [`graph4` (50n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/graph4_semantic_surprise.png), [`graph4` (100n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/graph4_semantic_surprise.png): Every node shows a massive update-1 spike (`0.0004–0.0025`) causing $\tau(t)$ to decay slowly over updates 2–50 and **drop vertically like a cliff at update 51** when update 1 leaves the 50-element deque. On `50n` and `100n`, the plot title hardcodes `"Nodes 0..9"` while actually rendering nodes in first-completion order (`65, 81, 77, ...` for 100n).
- [`graph2_convergence_3way.png` (10n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/graph2_convergence_3way.png) & [`graph2` (100n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/graph2_convergence_3way.png): GL and APSM curves oscillate widely across steps (consistent with pre-training merged evaluation and Adam reset), while several FL nodes climb upward monotonically on local non-IID test splits.
- [`graph6_mse_boxplot_3way.png` (100n)](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/graph6_mse_boxplot_3way.png): Degenerate flat bars because `N_SEEDS = 1`.
- [`generate_table_images.py`](file:///d:/coolage%20study%20materials/MTP/src/generate_table_images.py#L95-L97) renders `"packets saved"` / `"50.3%–54.2% broadcasts eliminated"` directly repeating `run_all_comparisons.py`'s flawed metric formula rather than physical packets saved (`9.2%–17.2%`).

---

## 4. Identified Limitations

### 4.1 Verified Limitations (Directly Confirmed in Code, Math & Artifacts)
1. **Communication / Bandwidth Metric Distortion:** Combining 5-second suppressed send attempts (`suppressed += 1`, no degree factor) with GL-Baseline neighbor-multiplied physical packets (`total_sent`) inflates reported savings (`50.3%–54.2%`) when actual transmitted packet reduction is **`9.19%`–`17.15%`**. Suppression also speeds up polling 6× (`5s` vs `30s`) and prolongs wall/sim time (`2540s` vs `2035s` in `phase2_run.log`).
2. **Statistical Flaw in Adaptive Threshold (`tau(t)`):** Pushing raw initial `val_loss` at update 1 into `_error_window` alongside subsequent tiny `|Δval_loss|` updates (`~100×` smaller) inflates `np.std` over updates 2–50 (`N=50`), generating artificial early suppression followed by a cliff drop at update 51 (when the threshold shrinks down near $\varepsilon(t)$ and the heartbeat begins forcing frequent transmissions). Surprise is also measured against static *lifetime best* validation loss (turning later updates into monotonic drift from a frozen minimum if global loss plateaus slightly higher), and uses Gaussian `2σ` bounds despite heavily skewed positive `|Δ|` values.
3. **Training & Synchronization Defects:**
   - **Pre-training evaluation:** [`Node.perform_update`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L204-L210) calls `_evaluate()` on `self._model` *before* [`SaveModelEvent`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L344-L351) installs `latest_weights`.
   - **Optimizer amnesia:** [`Node.train_model`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L233-L234) recreates the compiled model every round, resetting Adam moments on each 3-epoch step.
   - **Signal-vs-weight mismatch:** Surprise uses `best_update_val_loss`, but `save_model` / `marshal_model` transmit `latest_weights` (epoch 3).
   - **Stopped-node leakage:** [`process_send_model_event`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L155-L186) checks `FAILED` but not `STOPPED`, letting finished nodes keep sending stale final weights until all peers stop.
   - **Stale age-weighted asymmetry:** At `node_model_age == 0`, [`merge_weights_with_age_weighted_avg`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L87-L88) adds `0` to `ages_sum` while multiplying weights by `1`, and `new_model_age = max(...)` ignores actual freshness/staleness once `accumulated_weight` grows by `+21,946` per update.
4. **Scalability Degradation:** On `N=100`, APSM MSE degradation jumps to **`+6.64%`**, violating the repo's own `<= 5%` pass threshold ([`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L372)), even though `15,769` suppressions occurred.
5. **Reproducibility & Experimental Validity Failures:**
   - Single seed (`N_SEEDS = 1`, `std = 0`).
   - Asymmetric FL vs. GL/APSM test sets ([`federated_training.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/federated_training.py#L173-L198) vs [`extract_metrics`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L263-L295)).
   - Purely synthetic dataset contradicting "real Porto/Azure traces" claims in reports; broken [`threshold_sweep.py`](file:///d:/coolage%20study%20materials/MTP/src/threshold_sweep.py); zero energy implementation; missing raw `history.json`, missing `50n/100n` datasets & networks locally, and unreproducible doc numbers (`53.6%`, `49.61%`).

### 4.2 Potential Concerns (Requiring Further Runtime Inspection)
- **Model divergence / stale gossip under suppressed sparse graphs:** Because nodes suppress and poll every 5s while waiting neighbors stall or merge with stale/stopped peers, high-degree vs low-degree nodes may experience skewed update mixing (`events=4,205` vs `3,011` in `phase2_run.log`); verifying per-node merge frequency requires saved `history.json`.
- **Binary-encoded features + ReLU output ceiling (`Dense(1, activation='relu')`):** While data is scaled to `[0, 1]`, ReLU output can suffer dying neurons if peer averaging drives biases strongly negative, though not directly observed without per-node prediction dumps.
- **Fault-tolerance untested with APSM:** `failure_mode` (`NODE_FAILURE`, `LINK_FAILURE`) was never benchmarked alongside APSM gating; interaction between node recovery (`SendModelsLoopEvent(t+1)`) and suppressed heartbeat state is unverified.

---

## 5. Final Knowledge Assessment (A–F)

### A. APSM Architecture Summary
The implemented codebase extends Tundo et al.'s Keras/NetworkX discrete-event Gossip Learning simulator by inserting a lightweight scalar gating mechanism inside `Node` and `event.py`: after each local training update completes in `SaveModelEvent`, a node computes $\varepsilon(t) = |\text{best\_update\_val\_loss} - \text{lifetime\_best\_val\_loss}|$ (or raw loss on update 1), appends it to a `maxlen=50` deque, sets $\tau(t) = 2.0 \cdot \text{std}(\text{deque})$ for window length $\ge 2$, and checks $\varepsilon(t) \le \tau(t)$ before sending `latest_weights` in `SendModelsLoopEvent`, forcing a heartbeat broadcast after `5` consecutive suppressions and rescheduling suppressed nodes every `5s` instead of `30s`.

### B. Exact Source Files and Relevant Functions/Classes

| Component | Exact File | Classes / Functions |
|---|---|---|
| Config & APSM Hyperparameters | [`config.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/config.py#L53-L262) | `TrainingConfig` (`is_baseline`, `semantic_k`, `semantic_window`, `semantic_heartbeat`), `Config`, `HistoryConfig` |
| Central Scalability Config | [`experiment_config.py`](file:///d:/coolage%20study%20materials/MTP/src/experiment_config.py#L1-L84) | `N_NODES`, `K_EDGE_CONNECTIVITY`, `N_SEEDS=1`, `FIXED_UPDATES=100`, `SEMANTIC_K=2.0`, `SEMANTIC_WINDOW=50`, `SEMANTIC_HEARTBEAT=5` |
| Node Training, Eval & APSM Filter | [`node.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/node.py#L37-L414) | `Node`, `NodeState`, `update_semantic_state()`, `should_suppress_transmission()`, `perform_update()`, `train_model()`, `marshal_model()`, `save_model()`, `_evaluate()`, `persist_best_model()` |
| DES Event Loop & Gating Hook | [`event.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/event.py#L15-L359) | `process_send_model_event()`, `process_receive_model_event()`, `process_save_model_event()`, `TRAIN_TIME=5` |
| Simulator Runner & Post-Finetuning | [`simulator.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/simulator.py#L36-L206) | `Simulator.__init__()`, `Simulator.run_training_simulation()`, `_simulate_unreliable_link()` |
| Weight Aggregation & Marshaling | [`aggregators.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/aggregators.py#L56-L221), [`weights_marshaling.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/weights_marshaling.py#L12-L69) | `merge_weights_with_age_weighted_avg()`, `choose_aggregator()`, `marshal_weights_with_random_subsampling()`, `flatten_weights()`, `unflatten_weights()` |
| Telemetry & Weight Helpers | [`history.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/history.py#L9-L44), [`weight.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/weight.py#L6-L11), [`links_strategy.py`](file:///d:/coolage%20study%20materials/MTP/src/gossiplearning/links_strategy.py#L7-L30) | `History` (`suppressed_packets`, `semantic_surprise_scores`), `weight_by_dataset_size()`, `add_node_and_links_to_config()` |
| Model Definition | [`model_creators.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/model_creators.py#L10-L48) | `create_LSTM()` |
| Topology & Synthetic Data Pipeline | [`1_network_generation.py`](file:///d:/coolage%20study%20materials/MTP/src/1_network_generation.py#L29-L76), [`2_dataset_generation.py`](file:///d:/coolage%20study%20materials/MTP/src/2_dataset_generation.py#L86-L437), [`data.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/data.py#L11-L141) | `build_network()`, `get_tower_ideal_trends()`, `get_tower_real_trends()`, `encode_sequences_for_training()`, `train_val_test_split()`, `get_common_test_set()` |
| Benchmark Orchestrator & Plots | [`run_all_comparisons.py`](file:///d:/coolage%20study%20materials/MTP/src/run_all_comparisons.py#L126-L883), [`federated_training.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/federated_training.py#L30-L210), [`gossip_training.py`](file:///d:/coolage%20study%20materials/MTP/src/utils/gossip_training.py#L58-L104) | `run_seed()`, `run_one_simulation()`, `run_one_fl_simulation()`, `run_federated_simulation()`, `extract_metrics()`, `print_comparison_table()`, `plot_results()`, `save_results()`, `run_simulation()` |
| Unused / Broken Auxiliary Files | [`threshold_sweep.py`](file:///d:/coolage%20study%20materials/MTP/src/threshold_sweep.py#L62-L230), [`fl-baseline/`](file:///d:/coolage%20study%20materials/MTP/fl-baseline/fl_baseline/server_app.py#L22-L57), [`scalability_simulation.py`](file:///d:/coolage%20study%20materials/MTP/src/scalability_simulation.py#L21-L90), [`generate_table_images.py`](file:///d:/coolage%20study%20materials/MTP/src/generate_table_images.py#L14-L333) | `run_one_simulation()` (bugged `_SEMANTIC_K`), `CustomFedAvg`, `render_table()` |

### C. Results Table with Experiment Identifiers and Source Artifact Paths

| Exp ID | Scale | Method | Avg MSE (Scaled) | Pkts Sent | Suppressed | Reported Red. % | True Sent-Pkt Red. % | MSE Δ vs GL | Primary Source Artifact Path |
|---|---|---|---|---|---|---|---|---|---|
| `3way-10n-s0` | 10n | FL (local test*) | `0.021337` | `2,000` | `0` | — | — | — | [`results/phase2_full_10n/three_way_comparison_full.csv`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/three_way_comparison_full.csv) |
| `3way-10n-s0` | 10n | GL-Baseline | `0.031063` | `1,437` | `0` | `0.0%` | `0.0%` | `0.0%` | [`results/phase2_full_10n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/three_way_summary.json) |
| `3way-10n-s0` | 10n | APSM Phase 2 | `0.032376` | `1,305` | `1,702` | `54.22%` | **`9.19%`** | `+4.23%` | [`results/phase2_full_10n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_10n/three_way_summary.json) |
| `3way-50n-s0` | 50n | FL (local test*) | `0.005617` | `10,000` | `0` | — | — | — | [`results/phase2_full_50n/three_way_comparison_full.csv`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/three_way_comparison_full.csv) |
| `3way-50n-s0` | 50n | GL-Baseline | `0.010750` | `8,029` | `0` | `0.0%` | `0.0%` | `0.0%` | [`results/phase2_full_50n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/three_way_summary.json) |
| `3way-50n-s0` | 50n | APSM Phase 2 | `0.011083` | `6,652` | `8,112` | `50.26%` | **`17.15%`** | `+3.10%` | [`results/phase2_full_50n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_50n/three_way_summary.json) |
| `3way-100n-s0` | 100n | FL (local test*) | `0.005880` | `20,000` | `0` | — | — | — | [`results/phase2_full_100n/three_way_comparison_full.csv`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/three_way_comparison_full.csv) |
| `3way-100n-s0` | 100n | GL-Baseline | `0.007967` | `15,421` | `0` | `0.0%` | `0.0%` | `0.0%` | [`results/phase2_full_100n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/three_way_summary.json) |
| `3way-100n-s0` | 100n | APSM Phase 2 | `0.008495` | `13,240` | `15,769` | `50.56%` | **`14.14%`** | `+6.64%` | [`results/phase2_full_100n/three_way_summary.json`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full_100n/three_way_summary.json) |
| `kaggle-log-10n` | 10n | GL-Baseline | `0.00582` | `1,447` | `0` | `0.0%` | `0.0%` | `0.0%` | [`results/phase2_full/phase2_run.log`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full/phase2_run.log#L504-L522) |
| `kaggle-log-10n` | 10n | APSM Phase 2 | `0.00545` | `1,254` | `1,491` | `50.7%` | **`13.34%`** | `-6.34%` | [`results/phase2_full/phase2_run.log`](file:///d:/coolage%20study%20materials/MTP/results/phase2_full/phase2_run.log#L504-L522) |
| `doc-june30` *(unverified)* | 10n | GL vs APSM | `0.009269` vs `0.009332` | `1,447` vs `1,314` | `1,674` | `53.6%` | `9.19%` | `+0.69%` | [`README.md`](file:///d:/coolage%20study%20materials/MTP/README.md#L24-L33) / [`RESEARCH_REPORT.md`](file:///d:/coolage%20study%20materials/MTP/docs/RESEARCH_REPORT.md#L547-L558) *(backing CSV missing)* |
| `mtp1-script` *(unverified)* | 10n | GL vs APSM | `0.0318` vs `0.0332` | `3,204` vs `1,614` | `1,590` | `49.61%` | `49.63%` | `+4.50%` | [`presentation_script.md`](file:///d:/coolage%20study%20materials/MTP/presentation_script.md#L77-L82) *(no repo data/code)* |

### D. Confirmed Facts vs. Assumptions or Unverified Claims

| Category | Status | Details |
|---|---|---|
| APSM filter signal | **Confirmed** | Absolute change in 3-epoch `best_update_val_loss` vs lifetime `_best_val_loss`; **not** raw prediction error $|y_t - \hat{y}_t|$ (`node.py` L142-145). |
| Weights transmitted | **Confirmed** | Always `latest_weights` (epoch 3), while `_best_weights` is only saved at simulation end for `.h5` persistence (`node.py` L315, L352-370). |
| Dataset source | **Confirmed** | Synthetic Akima + AR(4) diurnal curves mapped to OpenCellID Porto cell coordinates (`2_dataset_generation.py`), **not** Porto GPS trajectories or Azure traces. |
| Bandwidth reduction % | **Confirmed Bug** | Reported `50.3%–54.2%` divides 5s decision-level suppressions by `(GL_messages_sent + suppressed)`; actual sent-packet savings vs GL-Baseline are **`9.19%–17.15%`**. |
| Update-1 spike & cliff drop | **Confirmed** | Update 1 pushes raw `val_loss` into `_error_window` (`deque(maxlen=50)`), inflating $\tau(t)$ for updates 2–50 and causing a cliff drop at update 51 (Graph 4). |
| FL vs GL fairness | **Confirmed Bug** | `federated_training.py` evaluates FL on local non-IID test splits into `nodes_test_history`; GL/APSM evaluate on `common_test_set` pre-training. |
| Energy savings (`Joules`) | **Unverified / Absent** | Zero energy model or Joule calculation exists anywhere in `src/`. |
| `305² = 93,025` grid scaling | **Unverified** | Used as a hardcoded display multiplier in `run_all_comparisons.py` L334, but synthetic datasets have no `305` spatial cell dimension in `2_dataset_generation.py`. |
| `0.009269 -> 0.009332` (`53.6%`) run & `threshold_sweep.py` curve | **Unverified / Missing** | Backing CSV/JSON (`results/Phase2_Full/`, `src/experiments_full/`) and sweep outputs (`results/phase 2/`) are absent; `threshold_sweep.py` has broken attributes and paths. |

### E. Missing Information That Must Be Resolved Before Phase 3
1. **Raw simulation telemetry (`history.json`, per-node `.h5` models, `config.json`)** for the `10n`, `50n`, and `100n` runs (directories inside gitignored `src/experiments/` and `exp-backup/` could not be listed due to workspace limits). Specifically needed to verify FL's `global_test_history`, exact `sim_t` finish times for 10n/50n/100n, and post-finetuning accuracy.
2. **Generated `50n` and `100n` data/topologies:** `data/datasets/porto_{50n,100n}_3k/` and `data/networks/porto_{50n,100n}_3k/` are missing from local disk. Reproducibility requires knowing which machine/script generated them (likely Kaggle).
3. **Provenance of `README.md` / `RESEARCH_REPORT.md` (`53.6%`) and MTP-1 PDF (`49.61%`, `100,200 J`):** Both `2022IMT082-MTP1(1).pdf` and `results/Phase2_Full/phase2_comparison_full.csv` are absent from the repository; if these numbers come from an external notebook or earlier commit, they need archiving or deprecation before Phase 3 papers/reports cite them.
4. **Canonical Phase 3 forecasting interface decision:** Since `node.py` saves `_best_weights` to `{id}.h5` after `finetuning_epochs=2`, whereas DES `eval_metrics` recorded pre-training `latest_weights`, Phase 3 offloading needs clarity on which model artifact represents the node's deployed forecaster and whether `data/datasets/` uses real traces or the synthetic generator.

### F. Confidence Assessment
- **Confidence level: High (~95%) in complete architectural and experimental mechanics; Moderate-Low (~40%) on missing external/gitignored artifacts.**
- **Reasons for high confidence:** Every line of the core simulation engine (`gossiplearning/*.py`), experiment runner (`run_all_comparisons.py`), FL implementation (`utils/federated_training.py`), dataset/network pipeline (`1_network_generation.py`, `2_dataset_generation.py`, `utils/data.py`), model factory (`utils/model_creators.py`), result summaries (`10n/50n/100n` CSVs/JSONs + `phase2_run.log`), and visual plots (`graph2`, `graph4`, `graph5`, `graph6`) was directly inspected and cross-checked mathematically. Every anomaly (pre-training `_evaluate()`, Adam reset, Update-1 outlier cliff at update 51, 5s vs 30s re-poll inflation, asymmetric FL test evaluation, `9.19%–17.15%` true packet savings) matches both code execution paths and visual plot signatures down to exact numbers.
- **Reasons for bounded confidence on uninspected state:** Gitignored directories (`src/experiments/`, `exp-backup/`, `src/configs/`, `src/scripts/`) could not be enumerated, and external documents/Kaggle outputs (`2022IMT082-MTP1(1).pdf`, `results/Phase2_Full/phase2_comparison_full.csv`, `50n/100n` `.npz`/`.txt` inputs) are not present on local disk.
