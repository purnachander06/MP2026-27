"""Robust line-by-line parser for Zeek/Bro conn.log.labeled files.

Handles:
- Dynamic header (#fields) discovery and column position indexing
- The IoT-23 nonstandard final field (21 tab columns with space-separated tail:
  tunnel_parents, label, detailed-label)
- Multi-word detailed labels
- Label normalization (case-insensitivity: benign/malicious)
- Explicit tracking and reporting of unknown labels (no silent drops)
- Missing value normalization ('-' and '(empty)')
- Malformed record detection and error diagnostics
- Data leakage prevention: explicit separation of ML features, metadata, and labels
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator
import re
from collections import Counter


# Explicit definition of fields to prevent data leakage:
# UIDs and IP addresses are strictly excluded from ML features.
METADATA_FIELDS = (
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "tunnel_parents",
    "scenario",
)

FEATURE_FIELDS = (
    "id.orig_p",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
)

TARGET_FIELDS = (
    "label",
    "detailed_label",
)

NUMERIC_FEATURE_FIELDS = (
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
)

CATEGORICAL_FEATURE_FIELDS = (
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
)


@dataclass
class HeaderInfo:
    """Stores parsed header metadata for a conn.log.labeled file."""
    fields: List[str]
    field_to_index: Dict[str, int]
    raw_header_line: str
    types: Optional[List[str]] = None


@dataclass
class ParsedRecord:
    """Represents a successfully parsed conn.log record."""
    # Metadata (EXCLUDED from ML features to prevent data leakage)
    scenario: str
    ts: Optional[float]
    uid: str
    orig_h: str
    resp_h: str
    tunnel_parents: str

    # ML Features
    orig_p: Optional[int]
    resp_p: Optional[int]
    proto: str
    service: str
    duration: Optional[float]
    orig_bytes: Optional[int]
    resp_bytes: Optional[int]
    conn_state: str
    local_orig: str
    local_resp: str
    missed_bytes: Optional[int]
    history: str
    orig_pkts: Optional[int]
    orig_ip_bytes: Optional[int]
    resp_pkts: Optional[int]
    resp_ip_bytes: Optional[int]

    # Targets
    label: str  # Normalized binary: 'benign' or 'malicious' (or unknown string)
    detailed_label: str  # Original detailed label
    raw_label: str

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert record to a dictionary."""
        d = {
            "id.orig_p": self.orig_p,
            "id.resp_p": self.resp_p,
            "proto": self.proto,
            "service": self.service,
            "duration": self.duration,
            "orig_bytes": self.orig_bytes,
            "resp_bytes": self.resp_bytes,
            "conn_state": self.conn_state,
            "local_orig": self.local_orig,
            "local_resp": self.local_resp,
            "missed_bytes": self.missed_bytes,
            "history": self.history,
            "orig_pkts": self.orig_pkts,
            "orig_ip_bytes": self.orig_ip_bytes,
            "resp_pkts": self.resp_pkts,
            "resp_ip_bytes": self.resp_ip_bytes,
            "label": self.label,
            "detailed_label": self.detailed_label,
        }
        if include_metadata:
            d["scenario"] = self.scenario
            d["ts"] = self.ts
            d["uid"] = self.uid
            d["id.orig_h"] = self.orig_h
            d["id.resp_h"] = self.resp_h
            d["tunnel_parents"] = self.tunnel_parents
        return d


class ZeekLogParser:
    """Streaming parser for Zeek/Bro labeled connection logs."""

    def __init__(self, scenario_name: str = "unknown"):
        self.scenario_name = scenario_name
        self.header_info: Optional[HeaderInfo] = None
        self.total_lines: int = 0
        self.comment_lines: int = 0
        self.valid_records: int = 0
        self.malformed_records: int = 0
        self.raw_labels_counter: Counter = Counter()
        self.normalized_labels_counter: Counter = Counter()
        self.detailed_labels_counter: Counter = Counter()
        self.unknown_labels_counter: Counter = Counter()
        self.malformed_samples: List[Tuple[int, str, str]] = []

    def parse_header_line(self, line: str) -> HeaderInfo:
        """Parse #fields header line into HeaderInfo."""
        stripped = line.strip()
        if not stripped.startswith("#fields"):
            raise ValueError(f"Line is not a valid #fields header: {line[:50]}")

        raw_fields = stripped[len("#fields"):].strip()
        fields = re.split(r"\s+", raw_fields)
        field_to_index = {field_name: idx for idx, field_name in enumerate(fields)}
        self.header_info = HeaderInfo(
            fields=fields,
            field_to_index=field_to_index,
            raw_header_line=line,
        )
        return self.header_info

    @staticmethod
    def normalize_label(raw_label: str) -> Tuple[str, bool]:
        """
        Normalize benign/malicious label capitalization.
        Returns:
            (canonical_label, is_known)
        Canonical labels: 'benign' and 'malicious'.
        If unknown, returns (raw_label, False).
        """
        cleaned = raw_label.strip()
        lowered = cleaned.lower()
        if lowered in ("benign", "normal"):
            return "benign", True
        elif lowered in ("malicious", "malware", "attack"):
            return "malicious", True
        return cleaned, False

    @staticmethod
    def _parse_int(val: str) -> Optional[int]:
        if not val or val in ("-", "(empty)"):
            return None
        try:
            return int(val)
        except ValueError:
            return None

    @staticmethod
    def _parse_float(val: str) -> Optional[float]:
        if not val or val in ("-", "(empty)"):
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @staticmethod
    def _clean_str(val: str, default: str = "-") -> str:
        if not val or val in ("-", "(empty)"):
            return default
        return val.strip()

    def parse_line(self, line: str, line_no: int = 0) -> Optional[ParsedRecord]:
        """
        Parse a single conn.log line into a ParsedRecord.
        Returns None for comments, empty lines, or malformed lines.
        """
        self.total_lines += 1
        stripped = line.strip()

        if not stripped:
            return None

        if stripped.startswith("#"):
            self.comment_lines += 1
            if stripped.startswith("#fields") and self.header_info is None:
                self.parse_header_line(line)
            return None

        if self.header_info is None:
            self.malformed_records += 1
            if len(self.malformed_samples) < 5:
                self.malformed_samples.append((line_no, line[:80], "Missing #fields header"))
            return None

        expected_count = len(self.header_info.fields)
        parts = line.rstrip("\r\n").split("\t")

        # IoT-23 nonstandard format: 21 tab columns where the 21st tab item
        # contains: tunnel_parents, label, detailed-label separated by whitespace.
        if len(parts) == 21 and expected_count == 23:
            tail = re.split(r"\s+", parts[-1].strip(), maxsplit=2)
            if len(tail) == 3:
                parts = parts[:-1] + tail
            elif len(tail) == 2:
                # E.g. tunnel_parents and label present, detailed-label missing
                parts = parts[:-1] + [tail[0], tail[1], "-"]
            else:
                self.malformed_records += 1
                if len(self.malformed_samples) < 5:
                    self.malformed_samples.append((line_no, line[:80], f"Invalid tail split count: {len(tail)}"))
                return None

        # Standard 23-column tab format
        if len(parts) != expected_count:
            self.malformed_records += 1
            if len(self.malformed_samples) < 5:
                self.malformed_samples.append((line_no, line[:80], f"Expected {expected_count} parts, got {len(parts)}"))
            return None

        idx = self.header_info.field_to_index

        raw_label = parts[idx["label"]] if "label" in idx else "-"
        self.raw_labels_counter[raw_label] += 1

        normalized_label, is_known = self.normalize_label(raw_label)
        self.normalized_labels_counter[normalized_label] += 1
        if not is_known:
            self.unknown_labels_counter[raw_label] += 1

        raw_detailed = parts[idx["detailed-label"]] if "detailed-label" in idx else "-"
        cleaned_detailed = self._clean_str(raw_detailed, default="-")
        self.detailed_labels_counter[cleaned_detailed] += 1

        record = ParsedRecord(
            scenario=self.scenario_name,
            ts=self._parse_float(parts[idx["ts"]]) if "ts" in idx else None,
            uid=self._clean_str(parts[idx["uid"]]) if "uid" in idx else "-",
            orig_h=self._clean_str(parts[idx["id.orig_h"]]) if "id.orig_h" in idx else "-",
            resp_h=self._clean_str(parts[idx["id.resp_h"]]) if "id.resp_h" in idx else "-",
            tunnel_parents=self._clean_str(parts[idx["tunnel_parents"]]) if "tunnel_parents" in idx else "-",
            orig_p=self._parse_int(parts[idx["id.orig_p"]]) if "id.orig_p" in idx else None,
            resp_p=self._parse_int(parts[idx["id.resp_p"]]) if "id.resp_p" in idx else None,
            proto=self._clean_str(parts[idx["proto"]], default="unknown") if "proto" in idx else "unknown",
            service=self._clean_str(parts[idx["service"]], default="-") if "service" in idx else "-",
            duration=self._parse_float(parts[idx["duration"]]) if "duration" in idx else None,
            orig_bytes=self._parse_int(parts[idx["orig_bytes"]]) if "orig_bytes" in idx else None,
            resp_bytes=self._parse_int(parts[idx["resp_bytes"]]) if "resp_bytes" in idx else None,
            conn_state=self._clean_str(parts[idx["conn_state"]], default="unknown") if "conn_state" in idx else "unknown",
            local_orig=self._clean_str(parts[idx["local_orig"]], default="-") if "local_orig" in idx else "-",
            local_resp=self._clean_str(parts[idx["local_resp"]], default="-") if "local_resp" in idx else "-",
            missed_bytes=self._parse_int(parts[idx["missed_bytes"]]) if "missed_bytes" in idx else None,
            history=self._clean_str(parts[idx["history"]], default="-") if "history" in idx else "-",
            orig_pkts=self._parse_int(parts[idx["orig_pkts"]]) if "orig_pkts" in idx else None,
            orig_ip_bytes=self._parse_int(parts[idx["orig_ip_bytes"]]) if "orig_ip_bytes" in idx else None,
            resp_pkts=self._parse_int(parts[idx["resp_pkts"]]) if "resp_pkts" in idx else None,
            resp_ip_bytes=self._parse_int(parts[idx["resp_ip_bytes"]]) if "resp_ip_bytes" in idx else None,
            label=normalized_label,
            detailed_label=cleaned_detailed,
            raw_label=raw_label,
        )

        self.valid_records += 1
        return record

    def parse_stream(self, lines: Iterator[str]) -> Iterator[ParsedRecord]:
        """Stream parsed records lazily without storing entire file in memory."""
        for line_no, line in enumerate(lines, start=1):
            rec = self.parse_line(line, line_no=line_no)
            if rec is not None:
                yield rec

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of parsing metrics."""
        return {
            "scenario": self.scenario_name,
            "total_lines": self.total_lines,
            "comment_lines": self.comment_lines,
            "valid_records": self.valid_records,
            "malformed_records": self.malformed_records,
            "raw_labels": dict(self.raw_labels_counter),
            "normalized_labels": dict(self.normalized_labels_counter),
            "detailed_labels": dict(self.detailed_labels_counter),
            "unknown_labels": dict(self.unknown_labels_counter),
            "malformed_samples": self.malformed_samples,
        }
