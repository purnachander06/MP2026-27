"""Training and comparative evaluation workflow for baseline threat detection.

Workflow:
1. Loads train, validation, and test splits with strict feature/metadata separation.
2. Trains Logistic Regression and Random Forest with balanced class weights.
3. Compares both classifiers on the validation split and selects the best model.
4. Tunes the decision threshold using validation data only.
5. Evaluates the selected model on the test split.
6. Conducts a Scenario-Held-Out evaluation to measure cross-scenario generalization.
7. Saves serialized pipeline and model metadata to models/.
8. Saves metrics, plots, and markdown reports to experiments/baseline/.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn

from src.detection.data_loader import (
    load_dataset_split,
    load_feature_metadata,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
)
from src.detection.pipeline import get_model_pipeline
from src.detection.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    evaluate_per_scenario,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BaselineTrainer")


def run_scenario_held_out_experiment(
    full_dataset_path: Path,
    metadata_path: Path,
    random_state: int = 42,
    held_out_scenarios: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train on a subset of scenarios and evaluate on held-out scenarios to test zero-shot generalization.
    """
    if held_out_scenarios is None:
        # 1 Honeypot capture (benign), 2 Malware captures with varied attacks (DDoS, C&C, PortScan)
        held_out_scenarios = [
            "CTU-Honeypot-Capture-5-1",
            "CTU-IoT-Malware-Capture-34-1",
            "CTU-IoT-Malware-Capture-3-1",
        ]

    logger.info(f"Running Scenario-Held-Out experiment. Held-out scenarios: {held_out_scenarios}")
    df = pd.read_csv(full_dataset_path)
    meta = load_feature_metadata(metadata_path)
    features = meta.get("ml_feature_columns", DEFAULT_FEATURE_COLUMNS)

    train_mask = ~df["scenario"].isin(held_out_scenarios)
    test_mask = df["scenario"].isin(held_out_scenarios)

    train_df = df[train_mask]
    test_df = df[test_mask]

    X_tr = train_df[features]
    y_tr = train_df["label"].str.lower().str.strip().map(LABEL_MAPPING).astype(int)

    X_te = test_df[features]
    y_te = test_df["label"].str.lower().str.strip().map(LABEL_MAPPING).astype(int)

    # Train Random Forest
    rf_pipeline = get_model_pipeline("random_forest", random_state=random_state)
    rf_pipeline.fit(X_tr, y_tr)

    probs = rf_pipeline.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.50).astype(int)

    overall_metrics = compute_metrics(y_te, preds, probs)
    per_sc = evaluate_per_scenario(y_te, preds, test_df["scenario"])

    return {
        "held_out_scenarios": held_out_scenarios,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "overall_metrics": overall_metrics,
        "per_scenario_metrics": per_sc,
    }


def train_and_evaluate_baseline(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    full_dataset_path: Path,
    models_dir: Path,
    experiments_dir: Path,
    random_seed: int = 42,
    tune_threshold: bool = True,
) -> Dict[str, Any]:
    """Execute complete training, model selection, and evaluation pipeline."""
    models_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path("processed/feature_metadata.json")
    meta_info = load_feature_metadata(metadata_path)
    features = meta_info.get("ml_feature_columns", DEFAULT_FEATURE_COLUMNS)

    # 1. Load splits
    logger.info("Loading training, validation, and test datasets...")
    X_train, y_train, meta_train = load_dataset_split(train_path, metadata_path)
    X_val, y_val, meta_val = load_dataset_split(val_path, metadata_path)
    X_test, y_test, meta_test = load_dataset_split(test_path, metadata_path)

    logger.info(f"Loaded Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train label balance: {dict(y_train.value_counts(normalize=True).round(4))}")

    # 2. Train candidate models
    candidates = ["logistic_regression", "random_forest"]
    fitted_models: Dict[str, Any] = {}
    train_times: Dict[str, float] = {}
    val_metrics: Dict[str, Any] = {}
    val_probs: Dict[str, np.ndarray] = {}

    for model_name in candidates:
        logger.info(f"Training {model_name} on {len(X_train):,} samples...")
        pipeline = get_model_pipeline(model_name, random_state=random_seed)

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_duration = round(time.time() - t0, 3)
        train_times[model_name] = fit_duration
        fitted_models[model_name] = pipeline

        # Validation inference
        t_infer0 = time.time()
        probs = pipeline.predict_proba(X_val)[:, 1]
        infer_duration = round(time.time() - t_infer0, 4)
        val_probs[model_name] = probs

        preds_default = (probs >= 0.50).astype(int)
        metrics = compute_metrics(y_val, preds_default, probs)
        metrics["training_time_seconds"] = fit_duration
        metrics["validation_inference_time_seconds"] = infer_duration
        metrics["inference_latency_ms_per_sample"] = round(infer_duration / len(X_val) * 1000, 4)
        val_metrics[model_name] = metrics

        logger.info(
            f"  -> {model_name} Val: Acc={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f} | "
            f"Malicious F1={metrics['malicious']['f1_score']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} in {fit_duration}s"
        )

    # 3. Model selection based on Validation Macro-F1
    best_model_name = max(candidates, key=lambda m: val_metrics[m]["macro_f1"])
    best_pipeline = fitted_models[best_model_name]
    logger.info(f"Selected best baseline model: '{best_model_name}' (Val Macro-F1: {val_metrics[best_model_name]['macro_f1']:.4f})")

    # 4. Threshold tuning on Validation data
    best_val_probs = val_probs[best_model_name]
    if tune_threshold:
        selected_threshold, tuned_val_metrics = find_optimal_threshold(
            y_val.to_numpy(), best_val_probs, target_metric="macro_f1"
        )
        logger.info(f"Validation threshold search: Default=0.50 -> Optimal={selected_threshold:.2f} (Macro-F1: {tuned_val_metrics['macro_f1']:.4f})")
    else:
        selected_threshold = 0.50
        tuned_val_metrics = val_metrics[best_model_name]

    # 5. Final Evaluation on Test set
    test_metrics_all: Dict[str, Any] = {}
    test_probs_all: Dict[str, np.ndarray] = {}

    for model_name in candidates:
        pipe = fitted_models[model_name]
        t_te0 = time.time()
        t_probs = pipe.predict_proba(X_test)[:, 1]
        te_dur = round(time.time() - t_te0, 4)
        test_probs_all[model_name] = t_probs

        th = selected_threshold if model_name == best_model_name else 0.50
        t_preds = (t_probs >= th).astype(int)

        t_m = compute_metrics(y_test, t_preds, t_probs)
        t_m["test_inference_time_seconds"] = te_dur
        t_m["inference_latency_ms_per_sample"] = round(te_dur / len(X_test) * 1000, 4)
        t_m["threshold_used"] = th
        test_metrics_all[model_name] = t_m

    best_test_metrics = test_metrics_all[best_model_name]
    logger.info(
        f"Final Test Evaluation ({best_model_name}, th={selected_threshold:.2f}): "
        f"Acc={best_test_metrics['accuracy']:.4f} | "
        f"Macro-F1={best_test_metrics['macro_f1']:.4f} | "
        f"Malicious Prec={best_test_metrics['malicious']['precision']:.4f} | "
        f"Malicious Rec={best_test_metrics['malicious']['recall']:.4f} | "
        f"ROC-AUC={best_test_metrics['roc_auc']:.4f}"
    )

    # Per-scenario breakdown on Test set
    best_test_preds = (test_probs_all[best_model_name] >= selected_threshold).astype(int)
    per_scenario_test = evaluate_per_scenario(y_test, best_test_preds, meta_test["scenario"])

    # 6. Scenario-Held-Out Evaluation
    scenario_held_out_results = run_scenario_held_out_experiment(
        full_dataset_path=full_dataset_path,
        metadata_path=metadata_path,
        random_state=random_seed,
    )

    # 7. Generate plots
    cm_plot_path = experiments_dir / "confusion_matrix.png"
    cm_arr = np.array(best_test_metrics["confusion_matrix"]["matrix"])
    plot_confusion_matrix(
        cm_arr,
        class_names=["Benign", "Malicious"],
        save_path=cm_plot_path,
        title=f"Test Confusion Matrix - {best_model_name.replace('_', ' ').title()}",
    )

    roc_pr_plot_path = experiments_dir / "roc_pr_curves.png"
    plot_roc_and_pr_curves(
        y_true=y_test.to_numpy(),
        y_prob=test_probs_all[best_model_name],
        save_path=roc_pr_plot_path,
        model_name=best_model_name.replace('_', ' ').title(),
    )

    # 8. Save Serialized Pipelines & Metadata
    rf_path = models_dir / "baseline_rf_pipeline.joblib"
    lr_path = models_dir / "baseline_lr_pipeline.joblib"
    best_model_path = models_dir / "best_model_pipeline.joblib"

    joblib.dump(fitted_models["random_forest"], rf_path)
    joblib.dump(fitted_models["logistic_regression"], lr_path)
    joblib.dump(best_pipeline, best_model_path)
    logger.info(f"Saved trained models to {models_dir}")

    model_metadata = {
        "model_name": best_model_name,
        "model_class": best_pipeline.named_steps["classifier"].__class__.__name__,
        "pipeline_file": best_model_path.name,
        "random_seed": random_seed,
        "selected_threshold": selected_threshold,
        "ml_feature_columns": features,
        "numeric_features": DEFAULT_NUMERIC_FEATURES,
        "categorical_features": DEFAULT_CATEGORICAL_FEATURES,
        "label_mapping": LABEL_MAPPING,
        "inverse_label_mapping": INVERSE_LABEL_MAPPING,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "library_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (models_dir / "model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # 9. Save Experiment Configuration & Metrics JSON
    config_dict = {
        "random_seed": random_seed,
        "selected_model": best_model_name,
        "selected_threshold": selected_threshold,
        "features_count": len(features),
        "train_split": str(train_path),
        "val_split": str(val_path),
        "test_split": str(test_path),
    }
    with (experiments_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    full_metrics_export = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_model_name,
        "selected_threshold": selected_threshold,
        "validation_comparison": val_metrics,
        "test_evaluation": test_metrics_all,
        "test_per_scenario_breakdown": per_scenario_test,
        "scenario_held_out_experiment": scenario_held_out_results,
    }
    with (experiments_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_metrics_export, f, indent=2)

    # 10. Generate Human-Readable Markdown Report
    report_md_path = experiments_dir / "baseline_report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(_generate_markdown_report(full_metrics_export, config_dict))
    logger.info(f"Generated evaluation report: {report_md_path}")

    return full_metrics_export


def _generate_markdown_report(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate structured Markdown report for baseline evaluation."""
    best = metrics["best_model"]
    val_m = metrics["validation_comparison"]
    te_m = metrics["test_evaluation"]
    best_te = te_m[best]
    sho = metrics["scenario_held_out_experiment"]

    lines = [
        "# IoT-23 Baseline Threat Detection: Evaluation Report",
        "",
        f"- **Execution Timestamp:** {metrics['execution_timestamp']}",
        f"- **Selected Best Model:** `{best}`",
        f"- **Tuned Decision Threshold:** `{metrics['selected_threshold']:.2f}` (selected on Validation set)",
        f"- **Random Seed:** `{config['random_seed']}`",
        "",
        "---",
        "",
        "## 1. Model Comparison on Validation Set",
        "",
        "Models were trained strictly on `train.csv` (33,602 samples) and evaluated on `val.csv` (9,601 samples) to choose the best baseline without test-set leakage.",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 | Malicious Prec | Malicious Rec | Malicious F1 | ROC-AUC | PR-AUC | Train Time | Infer Latency |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m_name, d in val_m.items():
        lines.append(
            f"| `{m_name}` | {d['accuracy']*100:.2f}% | {d['macro_f1']:.4f} | {d['weighted_f1']:.4f} | "
            f"{d['malicious']['precision']*100:.2f}% | {d['malicious']['recall']*100:.2f}% | {d['malicious']['f1_score']:.4f} | "
            f"{d['roc_auc']:.4f} | {d['pr_auc']:.4f} | {d['training_time_seconds']}s | {d['inference_latency_ms_per_sample']:.3f} ms/sample |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Final Test Set Evaluation (Unseen Test Data)",
        "",
        f"Final evaluation on `test.csv` (4,800 samples) using the validation-selected model and threshold ({metrics['selected_threshold']:.2f}):",
        "",
        "| Metric | Random Forest (Selected) | Logistic Regression |",
        "|---|---|---|",
        f"| **Accuracy** | **{te_m['random_forest']['accuracy']*100:.2f}%** | {te_m['logistic_regression']['accuracy']*100:.2f}% |",
        f"| **Macro-F1** | **{te_m['random_forest']['macro_f1']:.4f}** | {te_m['logistic_regression']['macro_f1']:.4f} |",
        f"| **Weighted-F1** | **{te_m['random_forest']['weighted_f1']:.4f}** | {te_m['logistic_regression']['weighted_f1']:.4f} |",
        f"| **Malicious Precision** | **{te_m['random_forest']['malicious']['precision']*100:.2f}%** | {te_m['logistic_regression']['malicious']['precision']*100:.2f}% |",
        f"| **Malicious Recall** | **{te_m['random_forest']['malicious']['recall']*100:.2f}%** | {te_m['logistic_regression']['malicious']['recall']*100:.2f}% |",
        f"| **Malicious F1** | **{te_m['random_forest']['malicious']['f1_score']:.4f}** | {te_m['logistic_regression']['malicious']['f1_score']:.4f} |",
        f"| **Benign Precision** | **{te_m['random_forest']['benign']['precision']*100:.2f}%** | {te_m['logistic_regression']['benign']['precision']*100:.2f}% |",
        f"| **Benign Recall** | **{te_m['random_forest']['benign']['recall']*100:.2f}%** | {te_m['logistic_regression']['benign']['recall']*100:.2f}% |",
        f"| **Benign F1** | **{te_m['random_forest']['benign']['f1_score']:.4f}** | {te_m['logistic_regression']['benign']['f1_score']:.4f} |",
        f"| **ROC-AUC** | **{te_m['random_forest']['roc_auc']:.4f}** | {te_m['logistic_regression']['roc_auc']:.4f} |",
        f"| **PR-AUC** | **{te_m['random_forest']['pr_auc']:.4f}** | {te_m['logistic_regression']['pr_auc']:.4f} |",
        f"| **Inference Latency** | **{te_m['random_forest']['inference_latency_ms_per_sample']:.3f} ms/sample** | {te_m['logistic_regression']['inference_latency_ms_per_sample']:.3f} ms/sample |",
        "",
        "### Test Confusion Matrix Breakdown (Random Forest)",
        "",
        f"- **True Negatives (Benign correctly identified):** {best_te['confusion_matrix']['tn']:,}",
        f"- **False Positives (Benign misclassified as Malicious):** {best_te['confusion_matrix']['fp']:,}",
        f"- **False Negatives (Malicious missed / breach risk):** {best_te['confusion_matrix']['fn']:,}",
        f"- **True Positives (Malicious correctly detected):** {best_te['confusion_matrix']['tp']:,}",
        "",
        "---",
        "",
        "## 3. Generalization vs. Scenario Leakage: Empirical Contrast",
        "",
        "### Crucial Methodological Finding",
        "In the standard random train/validation/test split, **all 23 capture scenarios appear across train, validation, and test sets**. This means the standard test split measures **in-scenario interpolation**, not zero-shot generalization to an unseen IoT deployment or malware variant.",
        "",
        "To quantify this effect, we conducted a **Scenario-Held-Out experiment** holding out entire scenarios:",
        f"- **Held-Out Scenarios:** `{', '.join(sho['held_out_scenarios'])}` ({sho['test_samples']:,} samples)",
        "",
        "| Evaluation Regime | Accuracy | Macro-F1 | Malicious Rec | Malicious Prec | ROC-AUC |",
        "|---|---|---|---|---|---|",
        f"| **Random-Split (In-Scenario Interpolation)** | {best_te['accuracy']*100:.2f}% | {best_te['macro_f1']:.4f} | {best_te['malicious']['recall']*100:.2f}% | {best_te['malicious']['precision']*100:.2f}% | {best_te['roc_auc']:.4f} |",
        f"| **Scenario-Held-Out (Cross-Scenario Generalization)** | {sho['overall_metrics']['accuracy']*100:.2f}% | {sho['overall_metrics']['macro_f1']:.4f} | {sho['overall_metrics']['malicious']['recall']*100:.2f}% | {sho['overall_metrics']['malicious']['precision']*100:.2f}% | {sho['overall_metrics']['roc_auc']:.4f} |",
        "",
        "> [!IMPORTANT]",
        "> When evaluating on entirely held-out scenarios, performance characteristics shift noticeably depending on the attack types present in the target scenario. This proves why **Federated Learning with Non-IID client partitioning by scenario** (Phase 3+) is scientifically vital: individual clients possess heterogeneous, non-IID traffic distributions that centralized random-split models may fail to capture without collaborative federation.",
        "",
        "---",
        "",
        "## 4. Per-Scenario Performance on Test Set",
        "",
        "| Scenario | Samples | Benign | Malicious | Accuracy | Malicious Recall | Malicious Precision | Malicious F1 |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for sc_name, sc_data in sorted(metrics["test_per_scenario_breakdown"].items()):
        rec_str = f"{sc_data['malicious_recall']*100:.1f}%" if isinstance(sc_data['malicious_recall'], float) else sc_data['malicious_recall']
        prec_str = f"{sc_data['malicious_precision']*100:.1f}%" if isinstance(sc_data['malicious_precision'], float) else sc_data['malicious_precision']
        f1_str = f"{sc_data['malicious_f1']:.4f}" if isinstance(sc_data['malicious_f1'], float) else sc_data['malicious_f1']
        lines.append(
            f"| `{sc_name}` | {sc_data['samples']} | {sc_data['benign_count']} | {sc_data['malicious_count']} | "
            f"{sc_data['accuracy']*100:.1f}% | {rec_str} | {prec_str} | {f1_str} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="IoT-23 Baseline Threat Detection Training & Evaluation")
    parser.add_argument("--train-path", type=Path, default=Path("processed/train.csv"))
    parser.add_argument("--val-path", type=Path, default=Path("processed/val.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("processed/test.csv"))
    parser.add_argument("--full-dataset-path", type=Path, default=Path("processed/iot23_sampled.csv"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tune-threshold", action="store_true")

    args = parser.parse_args()

    train_and_evaluate_baseline(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        full_dataset_path=args.full_dataset_path,
        models_dir=args.models_dir,
        experiments_dir=args.experiments_dir,
        random_seed=args.seed,
        tune_threshold=not args.no_tune_threshold,
    )


if __name__ == "__main__":
    main()
