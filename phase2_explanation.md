# Phase 2: Predictive-Semantic Filter — Conceptual Explanation

## The Problem with the Baseline (GL)

In the original Gossip Learning (GL-Baseline) setup, nodes are like students in a collaborative classroom. Every single time a student finishes a practice test (a training round), they **immediately yell their answers to everyone around them**.

They do this **blindly and constantly** — even if their answers haven't changed since the last round. This wastes a massive amount of network bandwidth. It's like sending 100 emails that each say "No updates here!"

---

## The Core Idea: Only Speak When You Have Something New to Say

Phase 2 teaches each node to **stay silent unless its update is genuinely surprising**. This is the "Semantic" part — the node only broadcasts when the semantic value of its update is high (i.e., it has something meaningful to add).

> **Key Insight:** If a node's local traffic is completely normal and its AI model is predicting it accurately, its weights aren't changing much. Sending those weights wastes bandwidth. But if traffic suddenly spikes and the model is caught off guard, *that* surprise is worth broadcasting!

---

## The Three-Part Mechanism

### 1. The Surprise Score (epsilon)

Every time a node completes a training round, we compute a surprise score:

    epsilon(t) = |current_val_loss - previous_best_val_loss|

- If the error is similar to last time -> surprise is near zero (boring)
- If the error jumps suddenly -> surprise is large (something new happened!)

### 2. The Adaptive Noise Threshold (tau)

The node keeps a sliding memory of its last 50 surprise scores. It asks: "How much do my errors normally wiggle?" — that is the Standard Deviation.

    tau(t) = k * sigma(epsilon[t-50 ... t])
             k = 2  (covers 95% of normal statistical noise)

This threshold is adaptive — it adjusts automatically. The choice of k=2 is proven empirically via the threshold sweep script (threshold_sweep.py).

### 3. The Gag Rule

Before sending, the node asks one question:
- **epsilon <= tau** (boring update): Stay silent, save bandwidth
- **epsilon > tau** (surprising update): Broadcast to neighbors

---

## Code Mapping

| Concept | File | Function |
|---------|------|----------|
| Surprise score + threshold | node.py | `update_semantic_state()` |
| Gag rule decision | node.py | `should_suppress_transmission()` |
| Gag rule enforced | event.py | `process_send_model_event()` |
| State updated after training | event.py | `process_save_model_event()` |
| Metrics persisted | history.py | `suppressed_packets`, `semantic_surprise_scores` |
| Comparison + plots | run_all_comparisons.py | Full comparison table + 3 graphs |

---

## Expected Results

| Metric | GL-Baseline | APSM Phase 2 | Goal |
|--------|-------------|--------------|------|
| Avg MSE | 0.0140 | ~0.0147 | Within 5% |
| Packets Suppressed | 0 | ~50 | >= 40% reduction |
