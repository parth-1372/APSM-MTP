# Stage 3 Workload Forecasting & APSM Semantic Communication: Statistical Review

**Repository:** `APSM-MTP`  
**Branch:** `phase3-continuation`  
**Commit:** `02ace48`  
**Date:** November 13, 2026  
**Auditor:** Senior Distributed Systems Researcher, Multi-Agent RL Researcher, Independent Code Auditor  

---

## 1. Overview and Retraction of Premature Equivalence Claims

In earlier documentation, the performance of the Stage 3 causal residual filter ($\text{APSM-H1}$) relative to the unsuppressed Gossip Learning baseline ($\text{GL Baseline}$) was characterized as "statistically proven equivalent in accuracy." 

This document formally qualifies that statement:
1. **The Observed Delta**: Across three independent seeds, the empirical mean Test MSE increased by **$+0.0718\%$** ($0.036621$ for APSM-H1 vs. $0.036595$ for GL Baseline).
2. **Statistical Qualification**: With $N=3$ random seeds, **statistical equivalence cannot be claimed as formally proven** under rigorous two-one-sided $t$-tests (TOST) without an excessively wide indifference margin $\epsilon_{\text{equiv}}$.
3. **Temporal Dependence**: Because edge workload traces exhibit strong autocorrelation, treating test time bins as independent samples violates basic statistical assumptions. Any fine-grained significance testing must account for temporal block dependence.
4. **The Core Valid Claim**: What *is* empirically established is that APSM-H1 achieves an average **$53.33\%$ reduction in model broadcast rounds** while maintaining test MSE within a tiny observed margin of $+0.0718\%$, overwhelmingly outperforming the Phase 2 validation-loss method ($23.97\%$ suppression, $+3.03\%$ MSE loss).

---

## 2. Rigorous Separation of Communication Metrics

To avoid the conflation of terminology identified in the audit, the following definitions are strictly adhered to:

| Metric Name | Symbol / Code Variable | Physical Definition | GL Baseline (Mean) | APSM-H1 (Mean) | Observed Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Forecasting Accuracy** | `test_mse`, `test_rmse` | Mean squared error on held-out test split $[1035, 1380)$ ($345\text{ bins} \times 6\text{ nodes}$) | $0.036595 \pm 0.000227$ | $0.036621 \pm 0.000245$ | **$+0.0718\%$ MSE** |
| **Model Broadcast Rounds** | `test_broadcast_rounds` | Number of 30-second gossip iteration windows where a node broadcasted its model parameters | $1{,}032.0$ rounds | $481.67$ rounds | **$53.33\%$ rounds avoided** |
| **Model Packets** | `test_packets_sent` | Total network transport packets sent over physical links ($k=2$ neighbor transmissions per round) | $2{,}064.0$ packets | $963.33$ packets | **$53.33\%$ packets saved** |
| **Model Bytes** | `test_model_bytes` | Total volume of serialized PyTorch weights transmitted ($56{,}920\text{ bytes}$ per packet) | $117.48\text{ MB}$ ($112.04\text{ MB}$ raw) | $54.83\text{ MB}$ ($52.29\text{ MB}$ raw) | **$53.33\%$ bytes saved** ($62.65\text{ MB}$) |
| **Suppression Decisions** | `test_suppressed_checks`| Number of times a node evaluated residual surprise $H_1 \le \tau$ and suppressed transmission | $0.0$ checks | $550.33$ checks | **$53.33\%$ suppression rate** |

> [!IMPORTANT]
> Because the edge topology is a 2-regular partial mesh ($k=2$ physical neighbors per node), the ratio of broadcast rounds to packets is constant ($1 : 2$). Therefore, a $53.33\%$ reduction in broadcast rounds translates directly into a $53.33\%$ reduction in both network packets and transmitted model bytes.

---

## 3. Seed-Level Recalculation & Paired Analysis

From the historical record in `results/phase3/stage3_forecast_gate_6n.json` (`full_hist_p2_std_k1.5`):

### 3.1 Raw Per-Seed Records

$$\begin{array}{lcccc}
\hline
\textbf{Seed} & \textbf{GL Baseline MSE} & \textbf{APSM-H1 MSE} & \textbf{Paired Difference } \Delta & \textbf{APSM-H1 Broadcasts} \\
\hline
\text{Seed 11} & 0.0367508 & 0.0367239 & -0.0000269 \; (-0.073\%) & 491 \; (52.42\% \text{ supp.}) \\
\text{Seed 22} & 0.0367597 & 0.0368567 & +0.0000970 \; (+0.264\%) & 471 \; (54.36\% \text{ supp.}) \\
\text{Seed 33} & 0.0362746 & 0.0362835 & +0.0000089 \; (+0.025\%) & 483 \; (53.20\% \text{ supp.}) \\
\hline
\textbf{Mean} & \mathbf{0.0365950} & \mathbf{0.0366214} & \mathbf{+0.0000263} \; (\mathbf{+0.0718\%}) & \mathbf{481.67} \; (\mathbf{53.33\%}) \\
\textbf{Sample Std} & \pm 0.0002267 & \pm 0.0002449 & \pm 0.0000639 & \pm 10.07 \\
\hline
\end{array}$$

### 3.2 Paired Difference Statistics ($N=3$)

1. **Mean Difference**: $\bar{\Delta} = +0.0000263$
2. **Sample Standard Deviation of Differences**: $s_\Delta = 0.0000639$
3. **Standard Error**: $\text{SE}_\Delta = \frac{0.0000639}{\sqrt{3}} \approx 0.0000369$
4. **95% Confidence Interval** ($t_{0.025, 2} = 4.303$):
   $$\text{CI}_{95\%} = 0.0000263 \pm (4.303 \times 0.0000369) = [-0.000132, \; +0.000185]$$
   In percentage terms relative to baseline GL MSE:
   $$\text{Relative CI}_{95\%} = [-0.36\%, \; +0.51\%]$$

### 3.3 Statistical Equivalence Testing (TOST Formulation)
To formally establish equivalence, the confidence interval must fall entirely within a predefined equivalence bound $[-\delta_{\text{equiv}}, +\delta_{\text{equiv}}]$.
- If the equivalence threshold is set to $\delta_{\text{equiv}} = 1.0\%$ MSE degradation ($0.000366$), the $95\%$ CI $[-0.36\%, +0.51\%]$ lies entirely within $[-1.0\%, +1.0\%]$.
- However, if the threshold is set strictly at $\delta_{\text{equiv}} = 0.25\%$, equivalence cannot be asserted at $\alpha=0.05$ with $N=3$ seeds.
- **Scientific Conclusion**: Rather than claiming "proven equivalence," the paper must state:
  > *"Across three random seeds, APSM-H1 reduced model broadcast volume by an average of $53.33\%$ while the observed mean test MSE remained within $+0.0718\%$ of unconstrained Gossip Learning (95% CI: $[-0.36\%, +0.51\%]$).*

---

## 4. Addressing Temporal Dependence in Time-Series Workloads

In distributed edge workload forecasting, the sequence of prediction errors $e_i(t) = y_i(t) - \hat{y}_i(t)$ across successive 30s bins $t = 1, \dots, 345$ is **not independently and identically distributed (non-i.i.d.)**.

### 4.1 Autocorrelation of Residuals
Trace analysis of Alibaba 2017 shows significant temporal autocorrelation:
$$\rho_k = \frac{\text{Cov}(e(t), e(t-k))}{\text{Var}(e(t))} > 0 \quad \text{for lags } k \in [1, 4]$$
This serial correlation causes naive bin-level $t$-tests to severely underestimate standard errors, yielding artificially low $p$-values.

### 4.2 Recommended Dependence-Aware Methodology for Stage 6
When reporting fine-grained statistical comparisons in Stage 6:
1. **Moving Block Bootstrap (MBB)**:
   Resample contiguous blocks of length $b = \lceil T^{1/3} \rceil = \lceil 345^{1/3} \rceil \approx 7\text{ bins}$ ($210\text{ seconds}$) with replacement (Künsch, 1989; Politis & Romano, 1994) to preserve the underlying Markovian covariance structure.
2. **Diebold-Mariano Test with Newey-West Correction**:
   When comparing loss differentials $d(t) = e_{\text{GL}}(t)^2 - e_{\text{APSM}}(t)^2$, use the Diebold-Mariano (1995) test with heteroskedasticity- and autocorrelation-consistent (HAC) variance estimators:
   $$\hat{V}_{\text{HAC}} = \hat{\gamma}_0 + 2 \sum_{k=1}^h \left( 1 - \frac{k}{h+1} \right) \hat{\gamma}_k$$

---

## 5. Summary of Corrected Stage 3 Findings

1. **Suppression Ratio**: $53.33\%$ reduction in model broadcast rounds, packets, and bytes is empirically robust across all tested seeds ($52.42\%$, $54.36\%$, $53.20\%$).
2. **Accuracy Impact**: Observed MSE delta is $+0.0718\%$ (Seed 11: $-0.07\%$, Seed 22: $+0.26\%$, Seed 33: $+0.02\%$).
3. **Statistical Qualification**: The observed difference is small and consistent with minimal accuracy loss, but formal equivalence claims require qualification given $N=3$ seeds and temporal autocorrelation.
4. **Comparison to Phase 2**: Stage 3 causal residual filtering ($\text{APSM-H1}$) is vastly superior to Phase 2's validation-delta mechanism, delivering more than double the communication savings ($53.33\%$ vs $23.97\%$) with $\approx 42\times$ smaller MSE degradation ($+0.07\%$ vs $+3.03\%$).

