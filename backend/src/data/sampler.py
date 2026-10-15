"""Reproducible streaming stratified reservoir sampling for IoT-23 scenarios.

Features:
- O(K) memory bound: never loads large files into memory.
- Bounded per-scenario caps so multi-gigabyte scenarios do not dominate.
- Stratified by class within scenario (preserves both benign and malicious traffic).
- Deterministic and reproducible with configurable random seed.
- Comprehensive recording of per-scenario and per-class sampling statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import random
from collections import Counter
from src.data.parser import ParsedRecord


@dataclass
class SamplingConfig:
    """Configuration for reproducible sampling."""
    random_seed: int = 42
    target_total_samples: int = 48003
    max_per_scenario: int = 2500
    target_train_samples: int = 33602
    target_val_samples: int = 9601
    target_test_samples: int = 4800
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10


class ScenarioReservoir:
    """Maintains a memory-bounded stratified reservoir for a single scenario."""

    def __init__(self, scenario_name: str, max_samples: int, rng: random.Random):
        self.scenario_name = scenario_name
        self.max_samples = max_samples
        self.rng = rng

        # Reservoirs partitioned by normalized label
        self.reservoirs: Dict[str, List[ParsedRecord]] = {
            "benign": [],
            "malicious": [],
            "other": [],
        }
        self.seen_counts: Dict[str, int] = {
            "benign": 0,
            "malicious": 0,
            "other": 0,
        }

    def _get_class_bucket(self, label: str) -> str:
        if label == "benign":
            return "benign"
        elif label == "malicious":
            return "malicious"
        return "other"

    def feed(self, record: ParsedRecord) -> None:
        """Feed a single record into the streaming reservoir with dynamic balance."""
        bucket = self._get_class_bucket(record.label)
        self.seen_counts[bucket] += 1
        seen_count = self.seen_counts[bucket]

        target_half = max(1, self.max_samples // 2)
        current_total = sum(len(v) for v in self.reservoirs.values())

        # If reservoir has available total capacity, just append
        if current_total < self.max_samples:
            self.reservoirs[bucket].append(record)
            return

        # Reservoir is full (current_total == self.max_samples)
        other_bucket = "malicious" if bucket == "benign" else "benign"
        other_len = len(self.reservoirs.get(other_bucket, []))
        this_len = len(self.reservoirs[bucket])

        # If this bucket is under-represented (< target_half) and other is over-represented (> target_half),
        # replace an element from the over-represented bucket
        if this_len < target_half and other_len > target_half:
            idx = self.rng.randint(0, other_len - 1)
            self.reservoirs[other_bucket].pop(idx)
            self.reservoirs[bucket].append(record)
            return

        # Otherwise, standard reservoir sampling within this bucket
        if this_len > 0:
            j = self.rng.randint(0, seen_count - 1)
            if j < this_len:
                self.reservoirs[bucket][j] = record

    def get_samples(self) -> List[ParsedRecord]:
        """Return combined samples from all class buckets."""
        combined: List[ParsedRecord] = []
        for bucket_records in self.reservoirs.values():
            combined.extend(bucket_records)
        return combined

    def get_stats(self) -> Dict[str, Any]:
        """Return seen vs retained counts for this scenario."""
        retained = {k: len(v) for k, v in self.reservoirs.items()}
        return {
            "scenario": self.scenario_name,
            "seen": dict(self.seen_counts),
            "retained": retained,
            "total_seen": sum(self.seen_counts.values()),
            "total_retained": sum(retained.values()),
        }


class StratifiedStreamingSampler:
    """Coordinates streaming sampling across all scenarios."""

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()
        self.rng = random.Random(self.config.random_seed)
        self.scenario_reservoirs: Dict[str, ScenarioReservoir] = {}
        self.overall_seen: Counter = Counter()
        self.overall_detailed: Counter = Counter()

    def get_scenario_reservoir(self, scenario_name: str) -> ScenarioReservoir:
        """Get or initialize reservoir for a given scenario."""
        if scenario_name not in self.scenario_reservoirs:
            # Each scenario receives an independent, reproducibly seeded RNG
            scenario_seed = self.rng.randint(0, 2**31 - 1)
            sc_rng = random.Random(scenario_seed)
            self.scenario_reservoirs[scenario_name] = ScenarioReservoir(
                scenario_name=scenario_name,
                max_samples=self.config.max_per_scenario,
                rng=sc_rng,
            )
        return self.scenario_reservoirs[scenario_name]

    def feed_record(self, record: ParsedRecord) -> None:
        """Feed record to appropriate scenario reservoir."""
        self.overall_seen[record.label] += 1
        self.overall_detailed[record.detailed_label] += 1
        res = self.get_scenario_reservoir(record.scenario)
        res.feed(record)

    def collect_all_samples(self) -> List[ParsedRecord]:
        """Collect and reproducibly shuffle all sampled records."""
        all_samples: List[ParsedRecord] = []
        for reservoir in self.scenario_reservoirs.values():
            all_samples.extend(reservoir.get_samples())

        # Shuffle deterministically
        self.rng.shuffle(all_samples)

        # If total exceeds target_total_samples, cap while maintaining representation
        if len(all_samples) > self.config.target_total_samples:
            all_samples = all_samples[: self.config.target_total_samples]

        return all_samples

    def split_samples(
        self, samples: List[ParsedRecord]
    ) -> Tuple[List[ParsedRecord], List[ParsedRecord], List[ParsedRecord]]:
        """
        Split samples into train, validation, and test partitions.
        Uses exact targets if total matches target_total_samples,
        otherwise uses proportional train/val/test ratios.
        """
        total = len(samples)
        if total == self.config.target_total_samples:
            n_train = self.config.target_train_samples
            n_val = self.config.target_val_samples
            n_test = total - n_train - n_val
        else:
            n_train = int(total * self.config.train_ratio)
            n_val = int(total * self.config.val_ratio)
            n_test = total - n_train - n_val

        train_set = samples[:n_train]
        val_set = samples[n_train : n_train + n_val]
        test_set = samples[n_train + n_val :]

        return train_set, val_set, test_set

    def get_summary_report(self) -> Dict[str, Any]:
        """Compile detailed sampling summary report."""
        per_scenario = {}
        total_sampled = 0
        sampled_classes: Counter = Counter()
        sampled_detailed: Counter = Counter()

        for sc_name, res in sorted(self.scenario_reservoirs.items()):
            stats = res.get_stats()
            per_scenario[sc_name] = stats
            total_sampled += stats["total_retained"]
            for bucket, records in res.reservoirs.items():
                sampled_classes[bucket] += len(records)
                for r in records:
                    sampled_detailed[r.detailed_label] += 1

        return {
            "random_seed": self.config.random_seed,
            "target_total_samples": self.config.target_total_samples,
            "max_per_scenario": self.config.max_per_scenario,
            "total_seen_records": sum(self.overall_seen.values()),
            "total_sampled_records": total_sampled,
            "seen_class_distribution": dict(self.overall_seen),
            "sampled_class_distribution": dict(sampled_classes),
            "seen_detailed_label_distribution": dict(self.overall_detailed),
            "sampled_detailed_label_distribution": dict(sampled_detailed),
            "per_scenario_statistics": per_scenario,
        }
