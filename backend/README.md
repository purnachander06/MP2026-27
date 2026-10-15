# LLMs meet Federated Learning for Scalable and Secure IoT Management
## Research-Grade IoT Anomaly and Malware Detection System

This repository implements the end-to-end architecture for **“LLMs meet Federated Learning for Scalable and Secure IoT Management”** based on the IoT-23 cybersecurity dataset.

---

## Final Stage Documentation & Evaluation Artifacts

For major project review, evaluation panels, and viva voce defense, please refer directly to the dedicated **Final Stage Documentation Suite** located in [`docs/final_stage/`](docs/final_stage/):

| Document | File Path | Purpose / Highlights |
| :--- | :--- | :--- |
| **Project Overview** | [`docs/final_stage/project_overview.md`](docs/final_stage/project_overview.md) | High-level problem statement, FL/GSFS motivation, LLM role, and system scope. |
| **Architecture & Workflow** | [`docs/final_stage/architecture_and_workflow.md`](docs/final_stage/architecture_and_workflow.md) | Full Mermaid diagrams, component mapping, persistence schema, and API contracts. |
| **Installation & Run Guide** | [`docs/final_stage/installation_and_run_guide.md`](docs/final_stage/installation_and_run_guide.md) | 26-step PowerShell and Docker Compose beginner guide with exact commands. |
| **Live Demo Runbook** | [`docs/final_stage/live_demo_runbook.md`](docs/final_stage/live_demo_runbook.md) | 5-to-7 minute step-by-step live demonstration script with pre-flight checklist. |
| **Project Presentation Script** | [`docs/final_stage/project_presentation_script.md`](docs/final_stage/project_presentation_script.md) | 8-to-10 min faculty presentation script + 2-min condensed elevator pitch. |
| **Viva Questions & Answers** | [`docs/final_stage/viva_questions_and_answers.md`](docs/final_stage/viva_questions_and_answers.md) | 32 comprehensive technical Q&As with mathematical and architectural depth. |
| **Final Experimental Results** | [`docs/final_stage/final_experimental_results.md`](docs/final_stage/final_experimental_results.md) | Baselines, multi-round FL, and systematic 9-configuration GSFS ablation metrics. |
| **Ablation Studies Analysis** | [`docs/final_stage/ablation_study.md`](docs/final_stage/ablation_study.md) | Deep-dive research analysis answering all 10 GSFS ablation research questions. |
| **Security Audit & Limitations**| [`docs/final_stage/security_and_limitations.md`](docs/final_stage/security_and_limitations.md) | Threat model, defensive testing results, and honest engineering disclosures. |
| **Security Test Results** | [`docs/final_stage/security_test_results.md`](docs/final_stage/security_test_results.md) | Automated security test verification log across 7 defensive test cases. |
| **Repository Audit** | [`docs/final_stage/repository_audit.md`](docs/final_stage/repository_audit.md) | Complete 15-category codebase inventory before and after final stage. |
| **Cleanup Manifest** | [`docs/final_stage/cleanup_manifest.md`](docs/final_stage/cleanup_manifest.md) | Deletion manifest, retained files log, and production mock removal audit. |

---

## Phase 1: Data Engineering & Preprocessing

The Phase 1 pipeline converts ~44 GB of raw Zeek `conn.log.labeled` files across 23 scenarios into clean, balanced, and leakage-free dataset partitions without loading full files into RAM.

- **Streaming $O(K)$ Memory Bounding:** Stratified reservoir sampling maintains bounded memory (< 50 MB RAM) across multi-gigabyte log streams.
- **Robust Zeek Parser:** Handles the nonstandard final column (21 tab columns with whitespace-separated `tunnel_parents`, `label`, and `detailed-label`).
- **Label Capitalization Normalization:** Standardizes honeypot (`benign`) and malware (`Benign`, `Malicious`) labels into canonical lowercase `benign` and `malicious`.
- **Anti-Domination Scenario Caps:** Caps scenarios at 2,500 samples so massive captures (e.g., Captures 17, 33, 39, 43) do not drown out small honeypot captures.
- **Strict Data Leakage Prevention:** Connection UIDs, raw IPs, timestamps, and tunnel metadata are strictly excluded from model features. `scenario` is retained strictly as metadata for Non-IID client partitioning.

---

## Phase 2: Baseline Threat Detection

Phase 2 builds a reproducible, scikit-learn based baseline threat detection system using `ColumnTransformer` pipelines.

### 1. Model Architecture & Preprocessing
- **Feature Schema (16 ML features):**
  - **Numeric (10):** `id.orig_p`, `id.resp_p`, `duration`, `orig_bytes`, `resp_bytes`, `missed_bytes`, `orig_pkts`, `orig_ip_bytes`, `resp_pkts`, `resp_ip_bytes`
  - **Categorical (6):** `proto`, `service`, `conn_state`, `local_orig`, `local_resp`, `history`
- **Leakage-Free Preprocessing:**
  - Numeric features: imputed with training median; scaled with `StandardScaler` for Logistic Regression.
  - Categorical features: imputed with constant string `'missing'`, encoded with `OneHotEncoder(handle_unknown='ignore')`. Unseen categories at inference time never crash the pipeline.
  - Preprocessors are fitted strictly on `train.csv` (33,602 samples).
- **Classifiers Evaluated:**
  1. **Logistic Regression:** Linear baseline with `class_weight='balanced'`.
  2. **Random Forest:** Ensemble baseline with `class_weight='balanced'`, 100 estimators, max depth 16.
- **Validation-Based Selection & Threshold Tuning:**
  - Evaluated on `val.csv` (9,601 samples).
  - Selected model: **Random Forest** (Validation Macro-F1: 0.9891 vs. 0.8807 for Logistic Regression).
  - Decision threshold tuned on validation data: **0.60** (optimizes malicious precision to 99.64% and Macro-F1 to 0.9886 on test set).

### 2. Measured Benchmark Performance (Test Set: 4,800 samples)

| Metric | Random Forest (Selected Baseline) | Logistic Regression |
|---|---|---|
| **Accuracy** | **98.90%** | 88.73% |
| **Macro-F1** | **0.9886** | 0.8772 |
| **Weighted-F1** | **0.9890** | 0.8835 |
| **Malicious Precision** | **99.64%** | 84.08% |
| **Malicious Recall** | **98.48%** | 99.72% |
| **Malicious F1** | **0.9906** | 0.9124 |
| **Benign Precision** | **97.86%** | 99.45% |
| **Benign Recall** | **99.49%** | 73.03% |
| **Benign F1** | **0.9867** | 0.8421 |
| **ROC-AUC** | **0.9990** | 0.9225 |
| **PR-AUC** | **0.9993** | 0.9346 |
| **Inference Latency** | **0.009 ms / sample** | 0.003 ms / sample |

### 3. Critical Methodological Finding: Scenario Leakage vs. Generalization

In the standard benchmark split (`train.csv`, `val.csv`, `test.csv`), **all 23 scenarios appear across all splits**. The test set therefore measures **in-scenario interpolation**, not zero-shot generalization to an unseen IoT deployment.

To quantify this, we conducted an empirical **Scenario-Held-Out evaluation** holding out entire scenarios (`CTU-Honeypot-Capture-5-1`, `CTU-IoT-Malware-Capture-34-1`, `CTU-IoT-Malware-Capture-3-1`):

| Evaluation Regime | Accuracy | Macro-F1 | Malicious Recall | Malicious Precision | ROC-AUC |
|---|---|---|---|---|---|
| **Random-Split (In-Scenario Interpolation)** | **98.90%** | **0.9886** | **98.48%** | **99.64%** | **0.9990** |
| **Scenario-Held-Out (Cross-Scenario Generalization)** | **74.83%** | **0.7229** | **56.85%** | **72.92%** | **0.8973** |

> [!IMPORTANT]
> The performance drop from 98.90% to 74.83% when testing on unseen IoT scenarios proves why centralized models suffer from domain shift in real-world IoT environments. This establishes the scientific justification for **Federated Learning with Non-IID client partitioning by scenario** in Phase 3.

---

## Phase 3: Federated Learning with FedAvg

Phase 3 builds a reproducible, tested Federated Learning implementation using PyTorch and Flower (`flwr`) to train an IoT threat detection neural network across decentralized, heterogeneous clients without centralizing raw traffic data.

### 1. Neural Network Model & Preprocessing Architecture
- **Model Architecture (`IoTThreatClassifier`):**
  - **Input:** 125 preprocessed features (10 standardized numeric features + 6 one-hot encoded categorical features).
  - **Hidden Layers:** Configurable feed-forward MLP: `Linear(125, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 32) -> ReLU() -> Dropout(0.1) -> Linear(32, 2)`.
  - **Parameters:** 10,210 float32 parameters (39.88 KB raw state dictionary payload).
  - **Loss & Optimizer:** Cross-Entropy Loss with PyTorch Adam (`lr=0.001`, `weight_decay=1e-5`).
- **Leakage Prevention & Feature Alignment:**
  - `ColumnTransformer` is fitted exclusively on the pooled training dataset (`train.csv`), then frozen.
  - Every client receives an identical 125-feature input representation with zero runtime missing-value or unseen-category errors.
  - Validation (`val.csv`) and test (`test.csv`) sets are strictly isolated from client training partitions.

### 2. Scenario-Aware Non-IID Partitioning
To reflect real-world IoT deployments where different devices capture distinct traffic profiles, training records from `train.csv` are partitioned across **5 clients** using scenario-aware bin packing:
- **Total records:** 33,602 training records assigned with zero record loss and zero client overlap.
- **Client Distributions:**
  - **Client 0:** 6,600 samples | 2,752 benign (41.7%), 3,848 malicious (58.3%) | Scenarios: 34-1, 48-1, 49-1, 7-1
  - **Client 1:** 6,746 samples | 2,593 benign (38.4%), 4,153 malicious (61.6%) | Scenarios: 1-1, 33-1, 4-1
  - **Client 2:** 6,756 samples | 2,476 benign (36.6%), 4,280 malicious (63.4%) | Scenarios: 20-1, 21-1, 3-1, 35-1, 44-1, 60-1
  - **Client 3:** 6,750 samples | 2,716 benign (40.2%), 4,034 malicious (59.8%) | Scenarios: 36-1, 39-1, 43-1, 9-1
  - **Client 4:** 6,750 samples | 2,465 benign (36.5%), 4,285 malicious (63.5%) | Scenarios: 17-1, 42-1, 5-1, 8-1

### 3. Federated Averaging (FedAvg) Workflow
1. **Server Parameter Broadcast:** Global server broadcasts weights $\theta_{\text{global}}$ to all participating clients.
2. **Local Client Optimization:** Each client $k$ initializes a local `IoTFlowerClient` (`NumPyClient`), trains locally on its private partition for $E=2$ epochs (`batch_size=64`, `Adam lr=0.001`).
3. **Parameter Upload:** Clients return updated weights $\theta_k$ and their local sample count $n_k$.
4. **Sample-Weighted Aggregation:** Global weights are updated via standard FedAvg:
   $$\theta_{\text{global}}^{(t+1)} = \sum_{k=1}^K \frac{n_k}{\sum_{j} n_j} \theta_k^{(t)}$$
5. **Global Validation:** Evaluated each round on `val.csv` (9,601 samples). Best global checkpoint is preserved for final test evaluation.

### 4. Fair 3-Way Benchmark Comparison (Test Set: 4,800 samples)

To isolate the effect of **federation** from the effect of **model architecture**, we compare the federated neural network against both the Phase 2 centralized Random Forest and an identically-architected centralized neural network:

| Metric | Centralized Random Forest (Phase 2) | Centralized Neural Network (20 Epochs) | Federated Neural Network (FedAvg, 5 Clients, 10 Rounds) |
|---|---|---|---|
| **Accuracy** | **98.90%** | 93.88% | 88.71% |
| **Macro-F1** | **0.9886** | 0.9354 | 0.8769 |
| **Weighted-F1** | **0.9890** | 0.9392 | 0.8833 |
| **Malicious Precision** | **99.64%** | 91.31% | 84.02% |
| **Malicious Recall** | 98.48% | 99.01% | **99.79%** |
| **Malicious F1** | **0.9906** | 0.9501 | 0.9123 |
| **Benign Precision** | 97.86% | 98.39% | **99.57%** |
| **Benign Recall** | **99.49%** | 86.64% | 73.08% |
| **Benign F1** | **0.9867** | 0.9208 | 0.8415 |
| **ROC-AUC** | **0.9990** | 0.9733 | 0.9358 |
| **PR-AUC** | **0.9993** | 0.9803 | 0.9567 |

#### Key Research Findings:
1. **Separating Federation vs. Architecture:** The drop from Centralized Random Forest (98.90%) to Centralized Neural Network (93.88%) is attributable to tree-based tabular advantages. The drop from Centralized NN (93.88%) to Federated FedAvg NN (88.71%) isolates the statistical penalty of decentralized Non-IID training.
2. **Prioritizing Attack Detection:** The FedAvg model achieves an outstanding **99.79% Malicious Recall** (only 6 missed malicious flows out of 2,829), ensuring malicious IoT traffic is almost never undetected.

### 5. Communication Payload Analysis
- **Model Parameters:** 10,210 float32 weights = 39.88 KB raw model payload.
- **Rounds & Updates:** 10 rounds $\times$ 5 participating clients = 50 total updates.
- **Total Payload Communicated:** 1.95 MB downloaded + 1.95 MB uploaded = **3.89 MB total**.
- **Execution Time:** Total client training time: 34.37s (~3.44s per round on CPU).
- *Note:* Payload calculation measures raw float32 parameter buffers without transport or serialization overhead. Standard FedAvg serves as the control baseline for Phase 4 communication-efficiency techniques.

---

## Phase 4: Federated Optimization with FedOpt (FedAdam)

Phase 4 evaluates **FedOpt**, a family of adaptive federated optimization strategies designed to address client drift and coordinate-wise gradient variance under heterogeneous, Non-IID client distributions.

### 1. Conceptual & Mathematical Foundation
In standard **FedAvg**, global updates are restricted to a direct weighted average of client weights ($\theta^{(t+1)} = \sum_{k=1}^K \frac{n_k}{N} \theta_k^{(t)}$). When clients train on disparate local distributions, client updates experience severe drift.

In **FedOpt** (Reddi et al., ICLR 2021), the server treats the aggregated parameter displacement as a **pseudo-gradient** for the global objective:
$$\Delta_t = \sum_{k=1}^K \frac{n_k}{N} \left(\theta_k^{(t)} - \theta^{(t)}\right)$$

The server then updates global parameters using an adaptive server optimizer. For **FedAdam**:
1. **First-Moment Momentum Vector:**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \Delta_t$$
2. **Second-Moment Variance Vector:**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) \Delta_t^2$$
3. **Bias-Corrected Learning Rate:**
   $$\eta_{\text{norm}} = \eta \frac{\sqrt{1 - \beta_2^{t+1}}}{1 - \beta_1^{t+1}}$$
4. **Adaptive Global Parameter Update:**
   $$\theta^{(t+1)} = \theta^{(t)} + \eta_{\text{norm}} \frac{m_t}{\sqrt{v_t} + \tau}$$

Where:
- $\eta$: Server learning rate (default: `0.02`, configurable)
- $\beta_1$: Momentum decay factor (default: `0.9`)
- $\beta_2$: Second-moment decay factor (default: `0.99`)
- $\tau$: Numerical stabilization parameter (default: `1e-3`, prevents division by zero / gradient explosion)

*(Note: Optional comparison strategies **FedYogi** and **FedAdagrad** are also supported via `--optimizer fedyogi` and `--optimizer fedadagrad`.)*

### 2. Comprehensive 4-Way Benchmark Comparison (Test Set: 4,800 Samples)

| Metric | Centralized Random Forest (Phase 2 Baseline) | Centralized Neural Network (Architectural Reference) | Federated FedAvg (Phase 3 Control) | Federated FedOpt (FedAdam, Phase 4) |
|---|---|---|---|---|
| **Accuracy** | **98.90%** | 93.88% | 88.71% | 88.71% |
| **Macro-F1** | **0.9886** | 0.9354 | 0.8769 | 0.8769 |
| **Weighted-F1** | **0.9890** | 0.9380 | 0.8832 | 0.8832 |
| **Malicious Precision** | **99.64%** | 91.31% | 84.02% | 84.02% |
| **Malicious Recall** | 98.48% | 99.01% | **99.79%** | **99.79%** |
| **Malicious F1** | **0.9906** | 0.9501 | 0.9123 | 0.9123 |
| **Benign Precision** | 97.86% | 98.39% | **99.59%** | **99.59%** |
| **Benign Recall** | **99.49%** | 86.54% | 72.87% | 72.87% |
| **Benign F1** | **0.9867** | 0.9208 | 0.8416 | 0.8416 |
| **ROC-AUC** | **0.9990** | 0.9733 | **0.9358** | 0.9028 |
| **Payload Communicated** | N/A (Centralized) | N/A (Centralized) | 3.89 MB | 3.89 MB |

#### Scientific Findings:
1. **Convergence Behavior:** FedAdam exhibits rapid initial acceleration in early rounds (reaching >87.5% validation accuracy in Round 2) due to momentum accumulation.
2. **Tabular Robustness:** On shallow tabular neural networks, FedAvg's direct parameter averaging provides strong intrinsic regularization. FedAdam matches FedAvg's accuracy (88.71%) and exceptional attack detection (**99.79% Malicious Recall**, missing only 6 attacks out of 2,824), with slightly lower ROC-AUC due to server momentum accumulation under extreme scenario skew.
3. **Communication Cost Invariance:** Because FedOpt operates strictly on the server-side aggregation step, communication bandwidth requirements remain identical to FedAvg (3.89 MB total communicated across 10 rounds $\times$ 5 clients).

---

## Phase 5: Federated Learning with GSFS (Gradient-Selective Federated Sampling)

Phase 5 implements **Gradient-Selective Federated Sampling (GSFS)**, an intelligent client-selection and update-management mechanism for federated IoT intrusion detection. Rather than blindly aggregating all updates or discarding updates from straggling or noisy edge devices, GSFS orchestrates an adaptive, four-component update-management pipeline:

### 1. Conceptual & Mathematical Foundation

```
[Simulated IoT Clients]
        │
        ├──> Client 0..4 Train Locally on Scenario Partitions
        │
[Server-Side GSFS Strategy Pipeline]
        │
        ├── 1. Performance-Change Trigger: ΔL = L_initial - L_final ≥ τ_perf
        ├── 2. Gradient-Statistics Trigger: ||Δθ||_2 ≥ μ_grad + β · σ_grad
        ├── 3. Trigger Combination: Select update if (Trigger_Perf OR/AND Trigger_Grad)
        ├── 4. Server Update Pool: Buffer updates (Bounded: max 10, Deduplication, Staleness ≤ 2)
        └── 5. Aggregation Threshold: If pool has ≥ ceil(60% × 5) = 3 updates:
                    θ_global ← θ_global + Σ (n_k / N_agg) · Δθ_k
```

1. **Performance-Change Trigger (`PerformanceTrigger`):**
   Evaluates whether a client's local training produced a statistically meaningful loss reduction on its private data partition:
   $$\Delta L = L_{\text{initial}} - L_{\text{final}}$$
   An update is flagged as performance-positive if $\Delta L \ge \tau_{\text{perf}}$ (default $\tau_{\text{perf}} = 0.0$).
2. **Gradient-Statistics Trigger (`GradientStatisticsTracker`):**
   Measures the Euclidean (L2) parameter displacement vector:
   $$\|\Delta\theta\|_2 = \sqrt{\sum_{l} \|\theta_{\text{local}}^{(l)} - \theta_{\text{global}}^{(l)}\|_2^2}$$
   Maintains the running fleet mean ($\mu_{\text{grad}}$) and sample standard deviation ($\sigma_{\text{grad}}$), evaluating against an adaptive threshold:
   $$\tau_{\text{grad}} = \mu_{\text{grad}} + \beta \cdot \sigma_{\text{grad}}$$
   where $\beta$ is a configurable selectivity multiplier (default: $\beta = 0.0$, selecting updates at or above average displacement).
3. **Server Update Pool (`UpdatePool`):**
   An in-memory, memory-bounded staging buffer (capacity: 10 models) that holds candidate parameter deltas $\Delta\theta_k$ along with metadata (client ID, round, model version, sample count). Guarantees:
   - **No Double-Application:** Updates are marked consumed upon aggregation and never applied twice.
   - **Staleness Expiration:** Stale updates exceeding `max_staleness` rounds (default: 2) are automatically pruned.
   - **Deduplication:** Fresher updates from a client replace pending older updates for that model version.
4. **Aggregation Threshold & Semantics:**
   The server triggers sample-weighted delta aggregation only when the pool contains at least:
   $$K_{\text{agg}} = \left\lceil \text{threshold} \times N_{\text{eligible}} \right\rceil = \lceil 0.60 \times 5 \rceil = 3 \text{ updates}$$
   When met, the global model updates via sample-weighted deltas:
   $$\theta_{\text{global}}^{(t+1)} = \theta_{\text{global}}^{(t)} + \sum_{k \in \mathcal{P}_{\text{agg}}} \frac{n_k}{\sum_{j \in \mathcal{P}_{\text{agg}}} n_j} \Delta\theta_k$$

---

### 2. Comprehensive 5-Way Benchmark Comparison (Test Set: 4,800 Samples)

| Metric | Centralized Random Forest (Phase 2) | Centralized Neural Network (Reference) | Federated FedAvg (Phase 3 Control) | Federated FedOpt (FedAdam, Phase 4) | Federated GSFS (Phase 5) |
|---|---|---|---|---|---|
| **Accuracy** | **98.90%** | 93.88% | 88.71% | 88.71% | **88.71%** |
| **Macro-F1** | **0.9886** | 0.9354 | 0.8769 | 0.8769 | **0.8769** |
| **Weighted-F1** | **0.9890** | 0.9380 | 0.8832 | 0.8832 | **0.8832** |
| **Malicious Precision** | **99.64%** | 91.31% | 84.02% | 84.02% | **84.02%** |
| **Malicious Recall** | 98.48% | 99.01% | **99.79%** | **99.79%** | **99.79%** |
| **Malicious F1** | **0.9906** | 0.9501 | 0.9123 | 0.9123 | **0.9123** |
| **Benign Precision** | 97.86% | 98.39% | **99.59%** | **99.59%** | **99.59%** |
| **Benign Recall** | **99.49%** | 86.54% | 72.87% | 72.87% | **72.87%** |
| **Benign F1** | **0.9867** | 0.9208 | 0.8416 | 0.8416 | **0.8416** |
| **ROC-AUC** | **0.9990** | 0.9733 | 0.9358 | 0.9028 | **0.9379** |
| **PR-AUC** | **0.9993** | 0.9795 | 0.9474 | 0.9260 | **0.9506** |
| **Payload Communicated** | N/A (Centralized) | N/A (Centralized) | 3.89 MB | 3.89 MB | 3.89 MB |

#### Scientific Findings:
1. **Intrusion Detection Performance Retention:** GSFS achieves identical classification accuracy (88.71%) and identical high attack sensitivity (**99.79% Malicious Recall**, missing only 6 out of 2,824 malicious flows) compared to FedAvg and FedOpt, confirming that selective buffer aggregation does not degrade cybersecurity threat detection.
2. **Superior Discrimination Quality:** GSFS achieves **ROC-AUC of 0.9379** and **PR-AUC of 0.9506**, outperforming both FedAvg (0.9358 ROC-AUC) and FedAdam (0.9028 ROC-AUC) on the held-out test set due to noise-resistant threshold aggregation.
3. **Selective Buffer Efficiency:** The 60% aggregation threshold ensures that the global model only updates when a quorum of representative client evidence is assembled.
4. **Communication Invariance:** In this simulation environment, all participating clients transmit updates to the server orchestrator so triggers can be evaluated, resulting in 3.89 MB total payload over 10 rounds. In an edge implementation with client-side trigger self-evaluation, only accepted updates transmit, yielding direct bandwidth savings.

---

## Windows PowerShell Quickstart Commands

All commands run from the workspace root using the virtual environment Python interpreter: `.\.venv\Scripts\python.exe`.

### 1. Run Complete Automated Test Suite (73 Tests)
Runs Phase 1 data tests (16), Phase 2 baseline detection tests (9), Phase 3 federated learning tests (14), Phase 4 FedOpt tests (12), and Phase 5 GSFS tests (22):

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

### 2. Run Phase 5 GSFS Federated Learning Experiment
Runs 5-client, 10-round GSFS simulation with 60% aggregation threshold, evaluates on test data, generates 5-way comparison bar charts, convergence curves, and selection dynamics plots:

```powershell
.\.venv\Scripts\python.exe -m src.federated.train_gsfs
```

### 3. Run Phase 5 Hyperparameter Sensitivity Grid
Runs controlled sensitivity tests across aggregation thresholds (40%, 60%, 80%), trigger modes (OR vs. AND), and gradient beta values:

```powershell
.\.venv\Scripts\python.exe -m src.federated.run_sensitivity
```

### 4. Run Phase 4 Federated Optimization Experiment (FedOpt / FedAdam)
```powershell
.\.venv\Scripts\python.exe -m src.federated.train_fedopt
```

### 5. Run Phase 3 Federated Learning Experiment (FedAvg)
```powershell
.\.venv\Scripts\python.exe -m src.federated.train
```

### 6. Run Phase 2 Baseline Training & Evaluation Workflow
```powershell
.\.venv\Scripts\python.exe -m src.detection.train
```

### 7. Run Preprocessing Pipeline (Phase 1)
```powershell
.\.venv\Scripts\python.exe -m src.data.pipeline --total-samples 48003 --max-per-scenario 2500 --seed 42
```

---

## Directory Structure & Generated Artifacts

```
├── dataset/                         # Original raw IoT-23 scenarios (unmodified)
├── processed/                       # Preprocessed dataset splits & metadata
│   ├── train.csv                    # 33,602 training samples
│   ├── val.csv                      # 9,601 validation samples
│   ├── test.csv                     # 4,800 test samples
│   ├── iot23_sampled.csv            # 48,003 complete samples
│   └── feature_metadata.json        # Schema contract & leakage prevention rules
├── models/                          # Serialized trained models & metadata
│   ├── best_model_pipeline.joblib   # Selected Random Forest pipeline (Phase 2)
│   ├── baseline_rf_pipeline.joblib  # Serialized Random Forest pipeline
│   ├── baseline_lr_pipeline.joblib  # Serialized Logistic Regression pipeline
│   ├── model_metadata.json          # Phase 2 model metadata
│   └── federated/                   # Federated models (Phases 3, 4, & 5)
│       ├── fedavg_model.pt          # PyTorch weights for best FedAvg model (Phase 3)
│       ├── fedavg_metadata.json     # Hyperparameters & metrics for FedAvg
│       ├── fedopt_model.pt          # PyTorch weights for best FedOpt model (Phase 4)
│       ├── fedopt_metadata.json     # Hyperparameters & metrics for FedOpt
│       ├── gsfs_model.pt            # PyTorch weights for best GSFS model (Phase 5)
│       └── gsfs_metadata.json       # Hyperparameters & metrics for GSFS
├── experiments/
│   ├── baseline/                    # Phase 2 Baseline evaluation artifacts
│   │   ├── baseline_report.md       # Comprehensive evaluation Markdown report
│   │   ├── metrics.json             # Complete machine-readable metrics export
│   │   ├── config.json              # Training & evaluation configuration
│   │   ├── confusion_matrix.png     # Test set confusion matrix visualization
│   │   └── roc_pr_curves.png        # ROC and Precision-Recall curves
│   ├── federated/fedavg/            # Phase 3 FedAvg artifacts
│   │   ├── report.md                # FedAvg research report
│   │   ├── metrics.json             # 3-way test set metrics
│   │   ├── config.json              # FedAvg configuration
│   │   ├── client_partitions.json   # Per-client scenario and class distribution
│   │   ├── round_metrics.csv        # Per-round validation metrics
│   │   ├── round_metrics.png        # FedAvg validation loss & accuracy curve
│   │   └── confusion_matrix.png     # FedAvg test set confusion matrix
│   ├── federated/fedopt/            # Phase 4 FedOpt artifacts
│   │   ├── report.md                # FedOpt audit research report
│   │   ├── metrics.json             # Full 4-way benchmark comparison metrics
│   │   ├── config.json              # FedOpt configuration
│   │   ├── client_partitions.json   # Per-client scenario and class distribution
│   │   ├── round_metrics.csv        # Per-round validation metrics and latency
│   │   ├── convergence.png          # Side-by-side FedAvg vs. FedOpt convergence curve
│   │   ├── comparison.png           # 4-way benchmark comparison bar chart
│   │   └── confusion_matrix.png     # FedOpt test set confusion matrix
│   ├── federated/gsfs/              # Phase 5 GSFS artifacts
│   │   ├── report.md                # GSFS comprehensive research report & sensitivity
│   │   ├── metrics.json             # Full 5-way benchmark comparison metrics
│   │   ├── config.json              # GSFS experiment configuration
│   │   ├── client_partitions.json   # Per-client scenario and class distribution
│   │   ├── round_history.csv        # Per-round validation metrics and pool lifecycle
│   │   ├── client_selection.csv     # Granular per-client trigger and selection log
│   │   ├── sensitivity_analysis.json# Empirical grid across thresholds, modes, and beta
│   │   ├── convergence.png          # 3-way FedAvg vs FedOpt vs GSFS convergence curve
│   │   ├── selection.png            # Client selection and gradient statistics dynamics
│   │   ├── comparison.png           # 5-way benchmark comparison bar chart
│   │   └── confusion_matrix.png     # GSFS test set confusion matrix
│   └── data_preprocessing_report.md # Phase 1 Data preprocessing report
├── src/
│   ├── data/                        # Phase 1 Data engineering modules
│   ├── detection/                   # Phase 2 Baseline detection modules
│   └── federated/                   # Federated learning modules (Phases 3, 4, & 5)
│       ├── __init__.py              # Package initialization
│       ├── config.py                # Hyperparameter configurations & dataclasses
│       ├── partitioner.py           # Scenario-aware Non-IID client partitioning
│       ├── model.py                 # PyTorch IoTThreatClassifier & parameter utils
│       ├── client.py                # Flower IoTFlowerClient (NumPyClient)
│       ├── strategy.py              # Sample-weighted FedAvg & FedOpt strategies
│       ├── server.py                # Federated coordinator & evaluation loop
│       ├── evaluate.py              # Evaluation metrics, 5-way table, and plot generators
│       ├── performance_trigger.py   # GSFS Performance-change trigger module
│       ├── gradient_statistics.py   # GSFS Gradient displacement & statistical tracker
│       ├── update_pool.py           # GSFS Server update buffer, deduplication & expiry
│       ├── gsfs_strategy.py         # Flower GSFSStrategy implementation
│       ├── train.py                 # Phase 3 FedAvg training entry point
│       ├── train_fedopt.py          # Phase 4 FedOpt training entry point
│       ├── train_gsfs.py            # Phase 5 GSFS training entry point
│       └── run_sensitivity.py       # Phase 5 Hyperparameter sensitivity grid
│   ├── explanations/                # Phase 6 Threat Explanation & LLM integration
│   │   ├── schemas.py               # Pydantic request/response schemas & severity levels
│   │   ├── prompt_builder.py        # Telemetry grounding & injection-resistant prompts
│   │   ├── providers/               # Base, Mock, Groq, and Gemini provider adapters
│   │   └── service.py               # Explanation service with deterministic fallback
│   ├── db/                          # Phase 7 PostgreSQL persistence & async repositories
│   │   ├── base.py                  # Declarative Base & cross-dialect JSON/UUID types
│   │   ├── models.py                # PredictionRecord & ExplanationRecord ORM definitions
│   │   ├── session.py               # Async engine, connection pooling, and health ping
│   │   └── repositories/            # Predictions, Explanations, and Analytics repositories
│   └── api/                         # Phase 6 & 7 FastAPI service
│       ├── main.py                  # FastAPI application entrypoint & middleware
│       ├── config.py                # Environment configuration settings
│       ├── dependencies.py          # Dependency injection & repository providers
│       └── routes/                  # Health, Predictions, Explanations, Analyze, History, Analytics
├── frontend/                        # Phase 8 React SOC Dashboard
│   ├── src/                         # React 19 source code (Vite + Tailwind CSS)
│   │   ├── components/              # Reusable UI cards, tables, charts, and layout
│   │   ├── pages/                   # Overview, Monitoring, Alerts, Models, Explanations, Analytics, Settings
│   │   ├── services/                # Axios API services matching FastAPI endpoints
│   │   └── utils/                   # Data formatters and constants
│   ├── tests/                       # Vitest & React Testing Library unit tests (19 tests)
│   ├── package.json                 # Frontend dependencies and run scripts
│   └── vite.config.js               # Vite configuration and proxy setup
├── alembic/                         # Phase 7 version-controlled database migrations
│   ├── env.py                       # Alembic async environment configuration
│   └── versions/                    # Initial Phase 7 persistence schema migration
└── tests/                           # Complete test suite (118 tests across Phases 1–7)
    ├── test_parser.py               # Tests for log parsing and header extraction (9 tests)
    ├── test_sampler.py              # Tests for streaming sampling bounds & reproducibility (6 tests)
    ├── test_pipeline.py             # Phase 1 pipeline integration test (1 test)
    ├── test_detection.py            # Phase 2 detection, leakage, and inference tests (9 tests)
    ├── test_federated.py            # Phase 3 federated partitioning, client, FedAvg tests (14 tests)
    ├── test_fedopt.py               # Phase 4 FedOpt strategies, sign, and server optimizer tests (12 tests)
    ├── test_gsfs.py                 # Phase 5 GSFS triggers, pool, threshold, and strategy tests (22 tests)
    ├── test_phase6_api_and_llm.py   # Phase 6 Adapter, Providers, Fallbacks, and API tests (17 tests)
    ├── test_phase7_database.py      # Phase 7 ORM models, repos, transactions, constraints (8 tests)
    ├── test_phase7_history.py       # Phase 7 Historical predictions, detail, pagination (11 tests)
    └── test_phase7_analytics.py     # Phase 7 SOC dashboard KPIs, time series, breakdown (9 tests)
```

---

## Phase 8: React SOC Dashboard for Federated IoT Threat Detection

Phase 8 implements a modern, research-grade Security Operations Center (SOC) dashboard using **React 19**, **Vite**, **Tailwind CSS**, **React Router 7**, **Recharts**, and **Axios**:

- **Cybersecurity SOC Dark Aesthetic:** Tailored dark navy color palette (`#0B1220` background, `#111C2E` panel, `#F26522` accent) providing readable, high-contrast visualization for security analysts without excessive decorative animations.
- **7 Fully Interactive Views:**
  1. **Overview Cockpit (`/`):** 5 live KPI cards, time-series alert trend charts, class breakdown, per-detector usage metrics, and recent alerts table.
  2. **Live Monitoring (`/monitoring`):** Near-real-time polling engine with configurable intervals (5s–30s), tab visibility lifecycle listeners, buffer counters, and pause/resume stream toggles.
  3. **Threat Alerts (`/alerts`):** Historical telemetry archive with server-side pagination, sorting allowlists, and multi-field filters (`predicted_class`, `detector_name`, `min_malicious_probability`).
  4. **Alert Details (`/alerts/:predictionId`):** Authoritative single-event inspection with uncalibrated score advisories, validated telemetry features, stripped data leakage exclusions, and on-demand LLM explanation generation.
  5. **Detection Models (`/models`):** Registry of 5 detector models (`random_forest`, `logistic_regression`, `fedavg`, `fedopt`, `gsfs`), live availability pings, and clear separation of offline 4,800-sample test-set benchmarks.
  6. **AI Explanations (`/explanations`):** Advisory LLM reasoning auditor with severity filters, fallback status tracking, human review flags, and structured investigation action plans.
  7. **Analytics (`/analytics`):** Historical aggregated analytics across volume, intrusion ratios, inference latency comparisons, and LLM explanation quality.
  8. **System Status (`/settings`):** Live FastAPI backend diagnostics, PostgreSQL readiness, and disk checkpoint verification.
- **Robust Client Architecture:** Centralized Axios client with automatic error normalization, friendly fallback states, loading skeletons, and zero unhandled rejections.
- **Automated Frontend Testing:** 19 Vitest unit tests verifying navigation, KPI cards, table pagination, filter forms, detail rendering, uncalibrated score notices, model availability, and streaming polling cleanup.
- **Complete Documentation:** Refer to `docs/phase8_soc_dashboard.md`.

### Quickstart: Running the Application

#### Terminal 1: FastAPI Backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2: React SOC Dashboard
```powershell
cd frontend
npm install
npm run dev
```
Open **`http://127.0.0.1:5173/`** in your browser.

#### Running Tests
- **Frontend Test Suite (Vitest):** `cd frontend; npm test` (19 passing)
- **Frontend Production Build:** `cd frontend; npm run build` (Clean exit code 0)
- **Backend Test Suite (Unittest):** `.\.venv\Scripts\python.exe -m unittest discover tests -v` (118 passing)


---

## Phase 6: LLM-Assisted Threat Explanations & FastAPI Service

Phase 6 implements a production-minded threat detection and LLM explanation layer:
- **Prediction Adapter:** Reusable adapter in `src/detection/adapter.py` supporting Centralized Random Forest, Centralized Logistic Regression, FedAvg, FedOpt, and GSFS with input validation and data leakage prevention.
- **LLM Reasoning Layer:** Grounded prompt builder and multi-provider architecture (`src/explanations/`) supporting a deterministic offline mock engine, Groq (Llama-3.3-70b-versatile), and Google Gemini (Gemini-1.5-Flash) with guaranteed deterministic fallback.
- **FastAPI Endpoints:** `GET /health`, `GET /ready`, `GET /models`, `POST /api/v1/predictions`, `POST /api/v1/explanations`, `POST /api/v1/analyze`.
- **Interactive Documentation:** Run the server (`uvicorn src.api.main:app`) and navigate to `http://127.0.0.1:8000/docs`. Detailed design documentation is located in `docs/phase6_llm_api.md`.

---

## Phase 7: PostgreSQL Persistence, Historical Threat Analytics & SOC Dashboard Backend

Phase 7 replaces the ephemeral in-memory cache with durable database persistence and provides a real-time historical analytics backend designed for modern Security Operations Center (SOC) dashboards:
- **Durable Relational Storage:** SQLAlchemy 2.x async ORM with `asyncpg` connection pooling persisting predictions, stripped leakage fields, model versions, and LLM advisory explanations into PostgreSQL (`predictions` and `explanations` tables).
- **Alembic Database Migrations:** Version-controlled database schema management (`alembic upgrade head` / `alembic downgrade -1`).
- **Database Readiness Health Checks:** Updated `GET /ready` executing active `SELECT 1` ping verification alongside model checkpoint inspections.
- **Historical Telemetry APIs:**
  - `GET /api/v1/history/predictions`: Server-side pagination, strict sorting allowlists, and multi-field filtering (detector, class label, confidence thresholds, date range).
  - `GET /api/v1/history/predictions/{prediction_id}`: Authoritative prediction lookup with all linked explanations.
  - `GET /api/v1/history/explanations`: Historical explanations with pagination, severity, and fallback status filters.
- **SOC Dashboard Analytics:**
  - `GET /api/v1/analytics/summary`: Real-time KPI summaries (total volume, malicious %, human review %, fallback %) with zero-division safety.
  - `GET /api/v1/analytics/alerts-over-time`: Chronological alert trend line buckets (`hour`, `day`, `week`).
  - `GET /api/v1/analytics/threat-breakdown`: Grouped threat distribution across detectors and classification outcomes.
  - `GET /api/v1/analytics/detectors`: Per-detector volume, class ratios, and average inference latency.
  - `GET /api/v1/analytics/explanations`: Explanation quality, severity breakdown, and LLM provider usage.
- **Security & Anti-Tampering:** Authoritative predictions are verified against the database before generating explanations, preventing clients from spoofing classifier outputs.
- **Detailed Documentation:** Refer to `docs/phase7_postgresql_persistence.md`.

---

## System Architecture

```mermaid
flowchart TD
    subgraph DataIngestion ["Phase 1: IoT-23 Raw Ingestion & Leakage-Free Preprocessing"]
        RAW["IoT-23 conn.log.labeled (23 Scenarios)"] --> PARSER["Robust Zeek Parser & Streaming Sampler"]
        PARSER --> CLEAN["Clean Datasets: train.csv (33,602), val.csv (9,601), test.csv (4,800)"]
        CLEAN --> PREPROC["Fitted ColumnTransformer (125 Features: 10 Scaled Numeric + 6 One-Hot Categorical)"]
    end

    subgraph TrainingRegimes ["Phases 2-5: Detection & Federated Learning Framework"]
        PREPROC --> BASELINE["Centralized Models (Random Forest, Logistic Regression, PyTorch MLP)"]
        PREPROC --> PARTITION["Scenario-Grouped Non-IID Partitioner (5 Virtual Clients)"]
        PARTITION --> CLIENTS["Decentralized IoTFlowerClients (Local Epochs = 2, Adam lr = 0.001)"]
        CLIENTS --> STRATEGIES{"Server Aggregation Strategy"}
        STRATEGIES -->|Phase 3| FEDAVG["FedAvg (Sample-Weighted Averaging)"]
        STRATEGIES -->|Phase 4| FEDOPT["FedOpt / FedAdam (Server Momentum & Adaptive Step)"]
        STRATEGIES -->|Phase 5| GSFS["GSFS (Performance & Gradient Trigger Selection + Update Pool)"]
    end

    subgraph ServingAndStorage ["Phases 6-7: Production Inference, Reasoning & Persistence"]
        FEDAVG & FEDOPT & GSFS & BASELINE --> DETECTOR["Trained Threat Classifier (PyTorch / scikit-learn)"]
        DETECTOR --> PREDICT["FastAPI Prediction Service (src/api/main.py)"]
        PREDICT --> DB[(PostgreSQL 18 Database: predictions & explanations)]
        PREDICT --> LLM_EXPLAIN["LLM Cybersecurity Reasoning (Mock / Groq / Gemini)"]
        LLM_EXPLAIN -.->|Advisory Context Only (Does NOT override detector)| DB
    end

    subgraph SOCDashboard ["Phases 8-9: Production Deployment & SOC Interface"]
        DB --> API_GATEWAY["API Gateway (FastAPI Uvicorn :8000)"]
        API_GATEWAY <--> NGINX["Nginx Web Server & Reverse Proxy (:5173 / :80)"]
        NGINX <--> REACT["React 19 + Vite SOC Dashboard (KPIs, Alert Triage, Explanations, Analytics)"]
    end
```

---

## Phase 9: Dockerization, Real PostgreSQL Validation & Final Experiments

Phase 9 completes the major research project by containerizing the end-to-end architecture, verifying durability against real PostgreSQL 18.3, and conducting comprehensive comparative experiments across all 6 intrusion detection paradigms.

### 1. Multi-Container Docker Architecture

The application is fully containerized using Docker Compose (`compose.yaml`):

| Service | Image / Build Context | Internal Port | Exposed Port | Role & Readiness Condition |
| :--- | :--- | :--- | :--- | :--- |
| **`db`** | `postgres:16-alpine` | `5432` | *(Internal only)* | Relational persistence with named volume `postgres_data`. Verified via `pg_isready -U postgres`. |
| **`api`** | `./Dockerfile` (Python 3.10 slim, `appuser`) | `8000` | `8000` | FastAPI backend. Runs `alembic upgrade head` at startup, health checked via `GET /ready`. Depends on `db: service_healthy`. |
| **`frontend`** | `./frontend/Dockerfile` (Multi-stage Node 20 builder + Nginx 1.25 runner) | `80` | `5173` | React SOC Dashboard. Serves SPA assets with `/index.html` fallback and reverse proxies `/api/`, `/health`, `/ready`, `/models` to API. Depends on `api: service_healthy`. |

### 2. Quickstart: Docker Compose Commands

#### Start the Full Stack
```powershell
docker compose up -d --build
```

#### Monitor Container Status and Health
```powershell
docker compose ps
```

#### View Service Logs
```powershell
docker compose logs -f api
docker compose logs -f frontend
docker compose logs -f db
```

#### Stop the System (Preserving Database Volume)
```powershell
docker compose down
```

#### Reset the Database (Destructive)
```powershell
docker compose down -v
```

### 3. Application Access Points
- **SOC Dashboard:** `http://localhost:5173/`
- **FastAPI Interactive API Docs:** `http://localhost:8000/docs`
- **Backend Readiness Probe:** `http://localhost:8000/ready`
- **Backend Liveness Probe:** `http://localhost:8000/health`
- **Model Checkpoints & Availability:** `http://localhost:8000/models`

---

## Complete Verification & Test Suite

| Test Suite | Execution Command | Result |
| :--- | :--- | :--- |
| **Backend Regression Suite** | `.\.venv\Scripts\python.exe -m unittest discover tests -v` | **118 passed** (100% pass rate) |
| **Real PostgreSQL 18 Suite** | `.\.venv\Scripts\python.exe -m unittest tests/test_phase9_real_postgres.py -v` | **9 passed** (Schema, FKs, Cascade Delete, CRUD) |
| **Frontend Unit Tests** | `cd frontend; npm test` | **19 passed** (7 test suites, 100% pass rate) |
| **Frontend Production Build** | `cd frontend; npm run build` | **Clean build** (`dist/` generated in 4.4s) |
| **Docker Compose Config** | `docker compose config` | **Valid configuration** (0 syntax or schema warnings) |

---

## Known Scientific & Operational Limitations

1. **Random Benchmark vs. Cross-Scenario Generalization**:
   - Random train/validation/test splits interpolate within known malware capture scenarios (achieving 98.9% Macro-F1 for Random Forest and 93.4% for neural models).
   - In cross-scenario evaluation where entire IoT environments are held out, performance drops significantly (74.8% for Random Forest, ~52-64% for neural models), reflecting real-world IoT feature distribution shift. Random split performance must NOT be interpreted as zero-shot generalization.
2. **Probability Calibration**:
   - Neural network softmax outputs and Random Forest leaf fractions are uncalibrated decision scores, not empirical event probabilities. Confidence advisories are displayed with disclaimers in the SOC UI.
3. **Simulated Federated Communication vs. Physical Client Suppression**:
   - In the Flower simulation environment, virtual clients transmit their updates to the server coordinator where GSFS triggers evaluate them. Theoretical bandwidth savings (bytes avoided) represent potential uplink conservation if triggers were executed client-side prior to transmission.
4. **LLM Explanations Are Strictly Advisory**:
   - The LLM reasoning module synthesizes context, extracts telemetry evidence, and suggests containment steps. It does NOT override, modify, or veto the underlying detector model's prediction.
5. **Security & Authentication Scope**:
   - Authentication (OAuth2/JWT) and Role-Based Access Control (RBAC) are not yet implemented in Phase 9. The dashboard and API must be deployed behind a secure VPN or reverse-proxy boundary in production environments.




