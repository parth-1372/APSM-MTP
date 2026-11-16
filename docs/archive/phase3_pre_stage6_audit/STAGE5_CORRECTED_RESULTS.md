# Stage 5 Corrected Experimental Results & Algorithmic Identification

**Repository:** `APSM-MTP`  
**Branch:** `phase3-continuation`  
**Commit:** `02ace48`  
**Date:** November 13, 2026  
**Auditor:** Senior Distributed Systems Researcher, Multi-Agent RL Researcher, Independent Code Auditor  

---

## 1. Finalized Stage 5 Algorithm Designation

Following rigorous code inspection of `src/phase3/marl/madqn.py`, `src/phase3/marl/mappo.py`, and `src/phase3/run_stage5_validation.py`, the reinforcement learning algorithms evaluated in Stage 5 are definitively classified as:

### 1.1 Corrected Nomenclature
- **B0**: `LocalOnly` (no offloading, local execution queue only).
- **E1**: `ReactiveGreedy` (heuristic greedy routing based on 15s periodic telemetry).
- **E2**: `Reactive Parameter-Shared Double Dueling Q-Learning` (Double Dueling MADQN without forecast augmentation).
- **E3**: `Forecast-Augmented Parameter-Shared Double Dueling Q-Learning` (Double Dueling MADQN with causal LSTM forecast feature).
- **O1**: `Privileged-Information Oracle Reference` (upper bound reference with exact future workload, zero telemetry delay, and link wire tracking).

> [!IMPORTANT]
> **Retraction of PPO/MAPPO Claims**:
> The earlier benchmark labels `e2_mappo` and `e2_ippo` did not evaluate PPO actor policies. In `run_stage5_validation.py` (lines 908 and 1058), the runner explicitly set `mode="q"`, which queried the parameter-shared Double Dueling Q-head (`q_loc`, `q_peer`) and completely bypassed the actor heads (`act_loc`, `act_peer`) and the centralized state critic (`v_glob`). All comparative claims attributing results to PPO or MAPPO are hereby formally retracted. The evaluated system is a Parameter-Shared Multi-Agent Double Dueling Q-Network.

### 1.2 Mathematical and Architectural Specification of the Evaluated Policy

1. **Dueling Q-Network Architecture**:
   State value and advantage streams are decoupled (Wang et al., 2016):
   $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}(s)|} \sum_{a' \in \mathcal{A}(s)} A(s, a') \right)$$
   where $V(s) \in \mathbb{R}$ is the scalar state value, $A(s, 0)$ is the local execution advantage, and $A(s, j)$ for $j \in \{1, \dots, K_{\text{max}}\}$ is the factored peer advantage.
2. **Double-DQN Target Updates**:
   To prevent overestimation bias (van Hasselt et al., 2016), target evaluation is decoupled from action selection:
   $$y_t = r_t + \gamma \cdot Q\left(s_{t+1}, \arg\max_{a} Q(s_{t+1}, a; \theta_{\text{online}}); \theta_{\text{target}}\right)$$
   with discount factor $\gamma = 0.40$ and smooth L1 loss ($\beta = 1.0$).
3. **Decentralized Parameter Sharing**:
   A single neural network parameterized by $\theta$ is shared across all $N=6$ edge nodes. At decision time, each node $i$ observes only its local decentralized observation vector $\mathbf{o}_i(t) \in \mathbb{R}^{75}$ (local queue, local CPU load, task compute/deadline requirements, and stale 15s neighbor beacons). No centralized state is consumed during inference.
4. **Action Masking**:
   Invalid offloading actions (full remote queue $Q_j \ge 48$, link failure, memory overflow, or tasks already offloaded once) are masked to $-\infty$ before computing Q-values or softmax distributions.
5. **Exploration vs. Evaluation**:
   - **Training Exploration**: Boltzmann temperature exploration with cosine schedule ($T_{\text{roll}} \in [0.012, 0.002]$) plus decaying $\epsilon$-greedy exploration ($\epsilon \in [0.30, 0.01]$).
   - **Evaluation**: Fully deterministic greedy action selection ($T = 0.0, \epsilon = 0.0$), choosing $a^* = \arg\max_{a \in \mathcal{A}(s)} Q(s, a)$.
6. **Proactive Forecast Branch in E3**:
   Arm E3 augments peer advantage with a proactive congestion residual:
   $$\mathbf{p}_{\text{mat}, j} = \left[ \max(0, \hat{y}_j(t+1) - \text{util}_j(t)), \; \hat{y}_j(t+1) \cdot \text{age}_j(t) \right]$$
   $$A_{\text{peer}}(s, j) = A_{\text{base}}(s, j) + \beta_{\text{fc}} \cdot \tanh(\mathbf{p}_{\text{mat}, j} \cdot \mathbf{w}_{\text{fc}}), \quad \text{with } \beta_{\text{fc}} = 0.008$$

---

## 2. Corrected Per-Seed Experimental Results

The raw metrics from `results/phase3/stage5_marl_gate_6n.json` were retrieved and independently recalculated. All task counts reconcile exactly with the held-out test workload ($M = 11{,}366\text{ tasks}$).

### 2.1 Complete Per-Seed Metric Breakdown

| Arm | Seed | Tasks Completed | Tasks Dropped | Drop Adm. | Drop Dead. | TSR (%) | DMR (%) | Eff. Latency Mean (s) | P95 Eff. Lat. (s) | Offload Rate (%) | Jain's Fairness | Total Comm. ($CB_{\text{total}}$) | Checkpoint Ep. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0 (LocalOnly)** | 11 | $8{,}578$ | $2{,}788$ | $343$ | $2{,}445$ | **$75.47\%$** | $24.53\%$ | $89.64\text{ s}$ | $443.8\text{ s}$ | $0.00\%$ | $0.9665$ | $0\text{ B}$ | N/A |
| | 22 | $8{,}578$ | $2{,}788$ | $343$ | $2{,}445$ | **$75.47\%$** | $24.53\%$ | $89.64\text{ s}$ | $443.8\text{ s}$ | $0.00\%$ | $0.9665$ | $0\text{ B}$ | N/A |
| | 33 | $8{,}578$ | $2{,}788$ | $343$ | $2{,}445$ | **$75.47\%$** | $24.53\%$ | $89.64\text{ s}$ | $443.8\text{ s}$ | $0.00\%$ | $0.9665$ | $0\text{ B}$ | N/A |
| **E1 (ReactiveGreedy)**| 11 | $9{,}328$ | $2{,}038$ | $109$ | $1{,}929$ | **$82.07\%$** | $17.93\%$ | $83.57\text{ s}$ | $408.6\text{ s}$ | $32.37\%$ | $0.9951$ | $323.82\text{ KB}$ | N/A |
| | 22 | $9{,}328$ | $2{,}038$ | $109$ | $1{,}929$ | **$82.07\%$** | $17.93\%$ | $83.57\text{ s}$ | $408.6\text{ s}$ | $32.37\%$ | $0.9951$ | $323.82\text{ KB}$ | N/A |
| | 33 | $9{,}328$ | $2{,}038$ | $109$ | $1{,}929$ | **$82.07\%$** | $17.93\%$ | $83.57\text{ s}$ | $408.6\text{ s}$ | $32.37\%$ | $0.9951$ | $323.82\text{ KB}$ | N/A |
| **E2 (Reactive MADQN)**| 11 | $9{,}852$ | $1{,}514$ | $51$ | $1{,}463$ | **$86.68\%$** | $13.32\%$ | $80.89\text{ s}$ | $415.0\text{ s}$ | $28.58\%$ | $0.9838$ | $309.50\text{ KB}$ | 40 |
| | 22 | $9{,}332$ | $2{,}034$ | $55$ | $1{,}979$ | **$82.10\%$** | $17.90\%$ | $85.48\text{ s}$ | $415.2\text{ s}$ | $33.71\%$ | $0.9820$ | $318.61\text{ KB}$ | 40 |
| | 33 | $9{,}754$ | $1{,}612$ | $74$ | $1{,}538$ | **$85.82\%$** | $14.18\%$ | $80.49\text{ s}$ | $400.0\text{ s}$ | $34.95\%$ | $0.9738$ | $320.81\text{ KB}$ | 44 |
| **E3 (Proactive MADQN)**| 11 | $9{,}852$ | $1{,}514$ | $51$ | $1{,}463$ | **$86.68\%$** | $13.32\%$ | $80.89\text{ s}$ | $415.0\text{ s}$ | $28.58\%$ | $0.9838$ | $309.50\text{ KB}$ | 40 |
| | 22 | $9{,}465$ | $1{,}901$ | $57$ | $1{,}844$ | **$83.27\%$** | $16.73\%$ | $84.86\text{ s}$ | $415.7\text{ s}$ | $32.96\%$ | $0.9782$ | $317.28\text{ KB}$ | 48 |
| | 33 | $9{,}843$ | $1{,}523$ | $69$ | $1{,}454$ | **$86.60\%$** | $13.40\%$ | $79.73\text{ s}$ | $395.4\text{ s}$ | $34.93\%$ | $0.9740$ | $320.78\text{ KB}$ | 44 |
| **O1 (Oracle Ref.)** | 11 | $9{,}983$ | $1{,}383$ | $38$ | $1{,}345$ | **$87.83\%$** | $12.17\%$ | $82.70\text{ s}$ | $425.8\text{ s}$ | $30.44\%$ | $0.9715$ | $54.06\text{ KB}$ | 48 |
| | 22 | $9{,}533$ | $1{,}833$ | $24$ | $1{,}809$ | **$83.87\%$** | $16.13\%$ | $84.74\text{ s}$ | $415.0\text{ s}$ | $34.22\%$ | $0.9735$ | $60.78\text{ KB}$ | 40 |
| | 33 | $9{,}844$ | $1{,}522$ | $39$ | $1{,}483$ | **$86.61\%$** | $13.39\%$ | $84.07\text{ s}$ | $418.6\text{ s}$ | $37.93\%$ | $0.9747$ | $67.36\text{ KB}$ | 44 |

---

## 3. Aggregate Statistical Analysis & Significance Evaluation

### 3.1 Mean and Sample Standard Deviation ($N=3$)

Unlike the previous audit table (which mistakenly duplicated aggregate means across seeds), the true seed-to-seed variability is reported below using sample standard deviation ($s = \sqrt{\frac{1}{N-1} \sum (x_i - \bar{x})^2}$):

$$\begin{array}{lccccc}
\hline
\textbf{Arm} & \textbf{TSR Mean (\%)} & \textbf{TSR Sample Std (\%)} & \textbf{DMR Mean (\%)} & \textbf{Eff. Lat. Mean (s)} & \textbf{Eff. Lat. Std (s)} \\
\hline
\text{B0 (LocalOnly)} & 75.4707\% & \pm 0.0000\% & 24.5293\% & 89.64\text{ s} & \pm 0.00\text{ s} \\
\text{E1 (ReactiveGreedy)} & 82.0693\% & \pm 0.0000\% & 17.9307\% & 83.57\text{ s} & \pm 0.00\text{ s} \\
\text{E2 (Reactive MADQN)} & 84.8671\% & \pm 2.4310\% & 15.1329\% & 82.29\text{ s} & \pm 2.78\text{ s} \\
\text{E3 (Proactive MADQN)} & \mathbf{85.5182\%} & \pm 1.9434\% & \mathbf{14.4818\%} & \mathbf{81.83\text{ s}} & \pm 2.69\text{ s} \\
\text{O1 (Oracle Reference)} & 86.1048\% & \pm 2.0272\% & 13.8952\% & 83.84\text{ s} & \pm 1.04\text{ s} \\
\hline
\end{array}$$

*Note: B0 and E1 are deterministic heuristics evaluated on fixed test arrivals; their seed variance is identically zero.*

### 3.2 Paired Differences: E3 (Proactive) vs. E2 (Reactive)

Because both E2 and E3 were evaluated on the identical test trace under identical seeds, the appropriate statistical unit of analysis is the **paired within-seed difference** $\Delta_s = \text{TSR}_{\text{E3}, s} - \text{TSR}_{\text{E2}, s}$:

- **Seed 11**: $\Delta_{11} = 86.6796\% - 86.6796\% = \mathbf{+0.0000\%}$ ($0$ additional tasks completed).
- **Seed 22**: $\Delta_{22} = 83.2747\% - 82.1045\% = \mathbf{+1.1702\%}$ ($+133$ additional tasks completed).
- **Seed 33**: $\Delta_{33} = 86.6004\% - 85.8173\% = \mathbf{+0.7830\%}$ ($+89$ additional tasks completed).

$$\text{Mean Paired Gain } \bar{\Delta} = \frac{0.0000 + 1.1702 + 0.7830}{3} = \mathbf{+0.6511\%}$$
$$\text{Sample Standard Deviation } s_\Delta = \sqrt{\frac{(0.0 - 0.6511)^2 + (1.1702 - 0.6511)^2 + (0.7830 - 0.6511)^2}{2}} = \mathbf{0.5966\%}$$
$$\text{Standard Error of the Mean } \text{SE}_\Delta = \frac{s_\Delta}{\sqrt{3}} = \frac{0.5966}{1.73205} = \mathbf{0.3445\%}$$

### 3.3 Statistical Significance and Confidence Intervals

1. **Paired $t$-Statistic**:
   $$t = \frac{\bar{\Delta}}{\text{SE}_\Delta} = \frac{0.6511}{0.3445} \approx \mathbf{1.890} \quad (df = 2)$$
   - Two-tailed $p$-value: **$p \approx 0.199$**
   - One-tailed $p$-value: **$p \approx 0.099$**
2. **95% Confidence Interval for $\bar{\Delta}$** ($t_{0.025, 2} = 4.303$):
   $$\text{CI}_{95\%} = 0.6511\% \pm (4.303 \times 0.3445\%) = [-\mathbf{0.830\%}, \; \mathbf{+2.133\%}]$$

> [!WARNING]
> **Scientific Qualification on Statistical Significance**:
> With only $N=3$ seeds ($df=2$), the $95\%$ confidence interval spans zero ($[-0.83\%, +2.13\%]$). Although the mean gain is positive (+0.65%, totaling $+222$ additional tasks completed across all runs) and no seed suffered performance degradation ($\Delta \ge 0$ for all seeds), **formal statistical significance cannot be claimed at the conventional $\alpha=0.05$ threshold with three seeds**. 
> Stage 6 must evaluate at least $N=5$ (preferably $N=10$) independent seeds to achieve statistical power.

---

## 4. Why Seed 11 Exhibited Zero Difference

A critical question is why Seed 11 produced identical results between E2 and E3 ($\Delta_{11} = 0.0000\%$).
- **Checkpoint Selection**: In Seed 11, the validation reward peaked at **Episode 40** (`best_vr = -0.8756`) and degraded on subsequent evaluation episodes (ep 44: `-0.8866`, ep 48: `-0.8853`).
- **Warmup Schedule**: In `run_stage5_validation.py`, `warmup = 36`. Forecast features were only active during episodes 37–40 (4 episodes of training).
- **Weight Magnitude**: At episode 40, the forecast branch weights had not yet grown large enough to overturn the argmax decision boundaries of the base network at temperature $T=0.0$.
- In contrast, in Seed 22, the validation reward peaked at **Episode 48** (after 12 episodes of forecast-augmented training), allowing the forecast branch to actively optimize routing decisions and deliver a $+1.17\%$ gain.

---

## 5. Summary of Corrected Claims

1. Stage 5 evaluated **Parameter-Shared Double Dueling MADQN**, not MAPPO or IPPO.
2. Across $3$ seeds, E3 improved over E2 by an average of **$+0.6511\%$ TSR**, saving a net total of **$222$ tasks** from dropping ($74$ tasks per seed on average).
3. The per-seed gains were: Seed 11: $+0.00\%$, Seed 22: $+1.17\%$, Seed 33: $+0.78\%$.
4. Claims of formal statistical significance are deferred to Stage 6 where $N \ge 5$ seeds will be trained.

