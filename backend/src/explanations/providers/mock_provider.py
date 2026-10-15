"""Deterministic mock explanation provider for offline testing and continuous integration."""

import time
from typing import Optional, Dict, Any, List
from src.explanations.providers.base import BaseExplanationProvider
from src.explanations.schemas import (
    SeverityLevel,
    ThreatExplanationResponse,
    ThreatPredictionResponse,
)


class MockExplanationProvider(BaseExplanationProvider):
    """
    Deterministic explanation provider generating structured, evidence-grounded
    cybersecurity explanations without external network or LLM dependencies.
    """

    def __init__(self, provider_name: str = "mock_threat_explainer_v1"):
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def is_available(self) -> bool:
        return True

    async def generate_explanation(
        self,
        prediction: ThreatPredictionResponse,
        operator_context: Optional[str] = None,
    ) -> ThreatExplanationResponse:
        """Generate deterministic explanation grounded in input prediction and telemetry."""
        evidence = prediction.validated_evidence
        is_malicious = prediction.predicted_class.lower() == "malicious"
        mal_prob = prediction.malicious_probability

        # Extract telemetry details for evidence construction
        proto = evidence.get("proto", "unknown")
        service = evidence.get("service", "generic")
        resp_p = evidence.get("id.resp_p", "unknown")
        orig_p = evidence.get("id.orig_p", "unknown")
        conn_state = evidence.get("conn_state", "unknown")
        duration = evidence.get("duration", 0.0)
        orig_pkts = evidence.get("orig_pkts", 0)
        resp_pkts = evidence.get("resp_pkts", 0)

        # 1. Determine severity and rationale
        if not is_malicious:
            severity = SeverityLevel.INFORMATIONAL
            severity_rationale = (
                f"Model classified flow as benign with {prediction.benign_probability:.2%} confidence. "
                f"Traffic characteristics align with expected protocol baselines."
            )
            summary = (
                f"The flow between source port {orig_p} and destination port {resp_p} over {proto.upper()} "
                f"demonstrates expected benign connection behavior (state: {conn_state}). "
                f"No malicious volumetric or behavioral anomalies were detected."
            )
            requires_human_review = False
        else:
            if mal_prob >= 0.90:
                severity = SeverityLevel.HIGH
                severity_rationale = (
                    f"High-confidence threat detection (malicious probability: {mal_prob:.2%}). "
                    f"Flow exhibits anomalous connection state ({conn_state}) on destination port {resp_p}."
                )
                summary = (
                    f"Potential intrusion or malicious automated activity detected over {proto.upper()} "
                    f"targeting port {resp_p} with state '{conn_state}'. High classifier confidence indicates "
                    f"behavioral alignment with known IoT attack patterns (e.g. C2 or scanning)."
                )
                requires_human_review = True
            else:
                severity = SeverityLevel.MEDIUM
                severity_rationale = (
                    f"Moderate-confidence threat detection (malicious probability: {mal_prob:.2%}, threshold: {prediction.threshold_used}). "
                    f"Connection exhibits non-standard telemetry deviations."
                )
                summary = (
                    f"Suspicious activity observed over {proto.upper()} to port {resp_p}. "
                    f"The model detected deviations from normal traffic baselines, but moderate confidence suggests "
                    f"possible benign operational anomalies."
                )
                requires_human_review = True

        # 2. Extract supporting evidence items
        supporting_evidence = [
            f"Observed transport protocol: {proto.upper()} with reported service '{service}'.",
            f"Endpoint port mapping: source port {orig_p} to destination port {resp_p}.",
            f"Connection termination state recorded as '{conn_state}' over {duration}s duration.",
            f"Packet transmission profile: {orig_pkts} originator packets and {resp_pkts} responder packets.",
        ]
        if operator_context:
            supporting_evidence.append(f"Noted operator context for triage reference: '{operator_context[:100]}'.")

        # 3. Formulate grounded uncertainties
        uncertainties = [
            "Deep packet payload inspection is unavailable; analysis is restricted to flow-level Zeek metadata.",
            "Potential false positive caused by legitimate IoT firmware telemetry polling or network latency.",
            f"Classifier probability ({mal_prob:.4f}) is an uncalibrated algorithmic score, not verified compromise proof.",
        ]

        # 4. Formulate low-risk, reversible defensive recommendations
        if is_malicious:
            recommended_actions = [
                f"Verify the authorized asset identity and expected business function of destination port {resp_p}.",
                f"Inspect host network logs for concurrent connection attempts from source port {orig_p} within +/- 15 minutes.",
                "Review firewall rules and determine whether this communication path matches baseline device policies.",
                "Capture targeted packet traces if the anomalous traffic pattern persists, without taking destructive remediation.",
            ]
        else:
            recommended_actions = [
                "No immediate defensive remediation required.",
                "Maintain standard telemetry monitoring and verify baseline adherence during scheduled audits.",
            ]

        return ThreatExplanationResponse(
            prediction_id=prediction.prediction_id,
            predicted_class=prediction.predicted_class,
            detector_name=prediction.detector_name,
            confidence_score=prediction.confidence_score,
            severity=severity,
            severity_rationale=severity_rationale,
            summary=summary,
            supporting_evidence=supporting_evidence,
            uncertainties=uncertainties,
            recommended_actions=recommended_actions,
            requires_human_review=requires_human_review,
            explanation_provider=self.provider_name,
            model_version=prediction.model_version,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            disclaimer="Advisory automated explanation grounded in flow telemetry. Not proof of compromise. Low-risk human triage required.",
            is_fallback=False,
        )
