"""Streaming Data Preprocessing Pipeline for IoT-23 Dataset.

Orchestrates:
1. Scenario discovery across all 23 captures
2. Streaming line-by-line parsing without loading full files into memory
3. Header validation, nonstandard final field extraction, label normalization
4. Unknown label tracking and reporting
5. Stratified reservoir sampling with configurable random seed and scenario caps
6. Train/Val/Test partitioning (e.g. 33,602 / 9,601 / 4,800)
7. Writing processed outputs strictly to processed/
8. Writing comprehensive reports strictly to experiments/ and docs/
9. Generating feature_metadata.json with strict data leakage prevention rules
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.data.parser import (
    ZeekLogParser,
    ParsedRecord,
    FEATURE_FIELDS,
    METADATA_FIELDS,
    TARGET_FIELDS,
    NUMERIC_FEATURE_FIELDS,
    CATEGORICAL_FEATURE_FIELDS,
)
from src.data.sampler import (
    StratifiedStreamingSampler,
    SamplingConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("IoT23Pipeline")


class PreprocessingPipeline:
    """Manages the full IoT-23 streaming preprocessing workflow."""

    def __init__(
        self,
        dataset_root: Path,
        processed_dir: Path,
        experiments_dir: Path,
        docs_dir: Path,
        config: SamplingConfig,
        max_lines_per_scenario: Optional[int] = None,
    ):
        self.dataset_root = dataset_root
        self.processed_dir = processed_dir
        self.experiments_dir = experiments_dir
        self.docs_dir = docs_dir
        self.config = config
        self.max_lines_per_scenario = max_lines_per_scenario

        self.sampler = StratifiedStreamingSampler(config=self.config)
        self.scenario_summaries: Dict[str, Dict[str, Any]] = {}
        self.pipeline_start_time: float = 0.0
        self.pipeline_end_time: float = 0.0

    def discover_scenario_files(self) -> List[Path]:
        """Find all conn.log.labeled files in dataset root."""
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root.resolve()}")

        files = sorted(self.dataset_root.glob("**/conn.log.labeled"))
        if not files:
            raise FileNotFoundError(f"No conn.log.labeled files found under: {self.dataset_root.resolve()}")
        return files

    def get_scenario_name(self, file_path: Path) -> str:
        """Extract the scenario directory name relative to dataset root."""
        rel_parts = file_path.relative_to(self.dataset_root).parts
        return rel_parts[0]

    def process_all_scenarios(self) -> None:
        """Stream-process all discovered scenarios."""
        files = self.discover_scenario_files()
        logger.info(f"Discovered {len(files)} scenario files.")
        self.pipeline_start_time = time.time()

        for idx, file_path in enumerate(files, start=1):
            sc_name = self.get_scenario_name(file_path)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"[{idx}/{len(files)}] Processing {sc_name} ({file_size_mb:.2f} MB)...")

            parser = ZeekLogParser(scenario_name=sc_name)
            sc_t0 = time.time()
            line_count = 0

            with file_path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_count += 1
                    if self.max_lines_per_scenario and line_count > self.max_lines_per_scenario:
                        break

                    record = parser.parse_line(line, line_no=line_count)
                    if record is not None:
                        self.sampler.feed_record(record)

            sc_dt = time.time() - sc_t0
            summary = parser.get_summary()
            summary["file_size_mb"] = round(file_size_mb, 2)
            summary["duration_seconds"] = round(sc_dt, 2)
            self.scenario_summaries[sc_name] = summary

            logger.info(
                f"  -> {sc_name}: Valid: {summary['valid_records']:,} | "
                f"Malformed: {summary['malformed_records']:,} | "
                f"Labels: {summary['normalized_labels']} in {sc_dt:.2f}s"
            )

        self.pipeline_end_time = time.time()
        logger.info(f"Streaming pass completed in {self.pipeline_end_time - self.pipeline_start_time:.2f}s")

    def _write_csv(self, records: List[ParsedRecord], output_path: Path) -> None:
        """Write records to CSV."""
        if not records:
            logger.warning(f"No records to write to {output_path}")
            return

        fieldnames = list(records[0].to_dict(include_metadata=True).keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict(include_metadata=True))

    def save_processed_datasets(self) -> Dict[str, int]:
        """Collect sampled records, partition, and save to processed directory."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Collecting and shuffling sampled records...")
        all_samples = self.sampler.collect_all_samples()
        total_samples = len(all_samples)
        logger.info(f"Total sampled records: {total_samples:,}")

        train_set, val_set, test_set = self.sampler.split_samples(all_samples)
        logger.info(
            f"Splits created: Train={len(train_set):,}, Val={len(val_set):,}, Test={len(test_set):,}"
        )

        full_csv = self.processed_dir / "iot23_sampled.csv"
        train_csv = self.processed_dir / "train.csv"
        val_csv = self.processed_dir / "val.csv"
        test_csv = self.processed_dir / "test.csv"

        logger.info(f"Writing {full_csv.name}...")
        self._write_csv(all_samples, full_csv)
        logger.info(f"Writing {train_csv.name}...")
        self._write_csv(train_set, train_csv)
        logger.info(f"Writing {val_csv.name}...")
        self._write_csv(val_set, val_csv)
        logger.info(f"Writing {test_csv.name}...")
        self._write_csv(test_set, test_csv)

        # Write feature metadata & data leakage prevention contract
        feature_metadata = {
            "dataset_name": "IoT-23 Preprocessed Sample",
            "random_seed": self.config.random_seed,
            "total_samples": total_samples,
            "splits": {
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "test_samples": len(test_set),
            },
            "ml_feature_columns": list(FEATURE_FIELDS),
            "numeric_feature_columns": list(NUMERIC_FEATURE_FIELDS),
            "categorical_feature_columns": list(CATEGORICAL_FEATURE_FIELDS),
            "target_columns": list(TARGET_FIELDS),
            "metadata_columns": list(METADATA_FIELDS),
            "data_leakage_prevention": {
                "excluded_columns": ["uid", "id.orig_h", "id.resp_h", "scenario", "ts", "tunnel_parents"],
                "rationale": {
                    "uid": "Excluded: unique connection ID causes model memorization and severe train/test overfitting",
                    "id.orig_h": "Excluded: raw source IP leaks network topology and device identities",
                    "id.resp_h": "Excluded: raw destination IP leaks external server identities",
                    "scenario": "Excluded from model inputs: reserved strictly as metadata for client partitioning and FL evaluation",
                    "ts": "Excluded from ML feature set: raw timestamp leaks temporal artifacts",
                    "tunnel_parents": "Excluded: Zeek tunneling metadata with no generalizable anomaly signal",
                },
                "enforcement": "ML models in Phase 2+ must strictly select ml_feature_columns as X and target_columns as y.",
            },
        }

        metadata_path = self.processed_dir / "feature_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Saved feature metadata to {metadata_path}")

        return {
            "total": total_samples,
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        }

    def generate_reports(self, split_counts: Dict[str, int]) -> None:
        """Generate comprehensive reports in experiments/ and docs/."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        sampler_summary = self.sampler.get_summary_report()
        total_time = self.pipeline_end_time - self.pipeline_start_time

        # Compile comprehensive metrics
        full_report = {
            "pipeline_version": "1.0.0",
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration_seconds": round(total_time, 2),
            "sampling_configuration": {
                "random_seed": self.config.random_seed,
                "target_total_samples": self.config.target_total_samples,
                "max_per_scenario": self.config.max_per_scenario,
                "train_samples": split_counts["train"],
                "val_samples": split_counts["val"],
                "test_samples": split_counts["test"],
            },
            "overall_summary": {
                "scenarios_processed": len(self.scenario_summaries),
                "total_seen_records": sampler_summary["total_seen_records"],
                "total_retained_samples": split_counts["total"],
                "seen_class_distribution": sampler_summary["seen_class_distribution"],
                "sampled_class_distribution": sampler_summary["sampled_class_distribution"],
                "seen_detailed_label_distribution": sampler_summary["seen_detailed_label_distribution"],
                "sampled_detailed_label_distribution": sampler_summary["sampled_detailed_label_distribution"],
            },
            "scenario_details": self.scenario_summaries,
            "sampler_scenario_stats": sampler_summary["per_scenario_statistics"],
        }

        # 1. Save JSON report to experiments/
        json_report_path = self.experiments_dir / "data_preprocessing_report.json"
        with json_report_path.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Saved JSON report to {json_report_path}")

        # 2. Save Markdown report to experiments/
        md_report_path = self.experiments_dir / "data_preprocessing_report.md"
        with md_report_path.open("w", encoding="utf-8") as f:
            f.write(self._build_markdown_report(full_report))
        logger.info(f"Saved Markdown report to {md_report_path}")

        # 3. Save Documentation artifact to docs/
        doc_path = self.docs_dir / "data_engineering.md"
        with doc_path.open("w", encoding="utf-8") as f:
            f.write(self._build_docs_page(full_report))
        logger.info(f"Saved documentation to {doc_path}")

    def _build_markdown_report(self, report: Dict[str, Any]) -> str:
        """Construct structured markdown report for experiments/."""
        lines = [
            "# IoT-23 Data Engineering & Preprocessing Report",
            "",
            f"- **Execution Time:** {report['execution_timestamp']}",
            f"- **Duration:** {report['total_duration_seconds']} seconds",
            f"- **Random Seed:** {report['sampling_configuration']['random_seed']}",
            f"- **Total Scenarios Processed:** {report['overall_summary']['scenarios_processed']}",
            f"- **Total Raw Records Seen:** {report['overall_summary']['total_seen_records']:,}",
            f"- **Total Retained Samples:** {report['overall_summary']['total_retained_samples']:,}",
            "",
            "## Dataset Splits",
            "",
            "| Partition | Sample Count | Percentage |",
            "|---|---|---|",
            f"| Train | {report['sampling_configuration']['train_samples']:,} | {report['sampling_configuration']['train_samples']/report['overall_summary']['total_retained_samples']*100:.2f}% |",
            f"| Validation | {report['sampling_configuration']['val_samples']:,} | {report['sampling_configuration']['val_samples']/report['overall_summary']['total_retained_samples']*100:.2f}% |",
            f"| Test | {report['sampling_configuration']['test_samples']:,} | {report['sampling_configuration']['test_samples']/report['overall_summary']['total_retained_samples']*100:.2f}% |",
            f"| **Total** | **{report['overall_summary']['total_retained_samples']:,}** | **100.00%** |",
            "",
            "## Label Distribution (Normalized Binary)",
            "",
            "| Label | Raw Seen Count | Sampled Retained Count | Sampled Ratio |",
            "|---|---|---|---|",
        ]

        seen_classes = report["overall_summary"]["seen_class_distribution"]
        sampled_classes = report["overall_summary"]["sampled_class_distribution"]
        total_sampled = report["overall_summary"]["total_retained_samples"]

        for lbl, count in sorted(sampled_classes.items(), key=lambda x: x[1], reverse=True):
            seen_cnt = seen_classes.get(lbl, 0)
            ratio = count / total_sampled * 100 if total_sampled else 0
            lines.append(f"| `{lbl}` | {seen_cnt:,} | {count:,} | {ratio:.2f}% |")

        lines.extend([
            "",
            "## Detailed Label Frequencies (Multi-Class)",
            "",
            "| Detailed Label | Raw Seen Count | Sampled Count | Percentage |",
            "|---|---|---|---|",
        ])

        seen_det = report["overall_summary"]["seen_detailed_label_distribution"]
        sampled_det = report["overall_summary"]["sampled_detailed_label_distribution"]

        for det_lbl, count in sorted(sampled_det.items(), key=lambda x: x[1], reverse=True):
            seen_cnt = seen_det.get(det_lbl, 0)
            ratio = count / total_sampled * 100 if total_sampled else 0
            lines.append(f"| `{det_lbl}` | {seen_cnt:,} | {count:,} | {ratio:.2f}% |")

        lines.extend([
            "",
            "## Per-Scenario Sampling Breakdown",
            "",
            "| Scenario | Total Seen | Benign Retained | Malicious Retained | Total Retained | Malformed Lines |",
            "|---|---|---|---|---|---|",
        ])

        for sc_name, sc_data in sorted(report["sampler_scenario_stats"].items()):
            seen_total = sc_data["total_seen"]
            retained_benign = sc_data["retained"].get("benign", 0)
            retained_mal = sc_data["retained"].get("malicious", 0)
            retained_tot = sc_data["total_retained"]
            malformed = report["scenario_details"].get(sc_name, {}).get("malformed_records", 0)
            lines.append(
                f"| `{sc_name}` | {seen_total:,} | {retained_benign:,} | {retained_mal:,} | {retained_tot:,} | {malformed:,} |"
            )

        return "\n".join(lines) + "\n"

    def _build_docs_page(self, report: Dict[str, Any]) -> str:
        """Construct docs/data_engineering.md."""
        return f"""# Phase 1: Data Engineering Architecture

This document describes the streaming preprocessing, normalization, sampling, and data leakage prevention pipeline built for the **IoT-23** dataset.

## 1. Overview & Research Context
- **Dataset:** IoT-23 (23 scenarios of labeled network traffic from real IoT devices and malware captures).
- **Goal:** Transform ~35+ GB of raw Zeek `conn.log.labeled` files into a balanced, leak-free, reproducible dataset of {report['overall_summary']['total_retained_samples']:,} samples for Federated Learning.
- **Random Seed:** {report['sampling_configuration']['random_seed']}

## 2. Key Engineering Solutions
1. **Streaming Memory Bounding ($O(K)$):**
   - No multi-gigabyte log file is ever loaded fully into RAM.
   - Files are parsed line-by-line using Python iterators.
   - Stratified reservoir sampling ensures constant memory consumption (< 50 MB RAM).

2. **Nonstandard Field Parsing:**
   - Zeek files in IoT-23 have 21 tab columns where the 21st column contains `tunnel_parents`, `label`, and `detailed-label` space-separated.
   - Handled with `re.split(r'\\s+', parts[-1].strip(), maxsplit=2)` to preserve multi-word detailed labels.

3. **Label Capitalization Normalization:**
   - Honeypot captures (`benign`) and malware captures (`Benign`) normalized to canonical lowercase `benign`.
   - `Malicious` normalized to `malicious`.
   - Unknown labels are recorded and reported rather than silently discarded.

4. **Strict Data Leakage Prevention:**
   - Excluded from ML features: `uid`, `id.orig_h` (source IP), `id.resp_h` (destination IP), `ts`, `tunnel_parents`.
   - Preserved as metadata: `scenario` is kept in dataset records for Federated Learning non-IID client assignment, but explicitly excluded from model training features in `processed/feature_metadata.json`.

5. **Anti-Domination Stratified Sampling:**
   - Per-scenario caps ({report['sampling_configuration']['max_per_scenario']}) prevent multi-gigabyte captures from overwhelming smaller honeypot captures.
   - Stratified buckets preserve minority classes in each scenario.
"""


def main() -> None:
    """CLI entrypoint for preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="IoT-23 Streaming Preprocessing Pipeline")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"dataset\opt\Malware-Project\BigDataset\IoTScenarios"),
        help="Root directory containing IoT-23 scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory to write processed output CSVs and metadata",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Directory to write audit reports",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory to write documentation",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=48003,
        help="Target total sample count (default: 48,003)",
    )
    parser.add_argument(
        "--max-per-scenario",
        type=int,
        default=2500,
        help="Maximum samples per scenario to prevent domination (default: 2,500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--max-lines-per-scenario",
        type=int,
        default=None,
        help="Optional limit on lines read per scenario for fast testing/dry run",
    )

    args = parser.parse_args()

    config = SamplingConfig(
        random_seed=args.seed,
        target_total_samples=args.total_samples,
        max_per_scenario=args.max_per_scenario,
    )

    pipeline = PreprocessingPipeline(
        dataset_root=args.dataset_root,
        processed_dir=args.output_dir,
        experiments_dir=args.experiments_dir,
        docs_dir=args.docs_dir,
        config=config,
        max_lines_per_scenario=args.max_lines_per_scenario,
    )

    pipeline.process_all_scenarios()
    counts = pipeline.save_processed_datasets()
    pipeline.generate_reports(counts)
    logger.info("Pipeline executed successfully!")


if __name__ == "__main__":
    main()
