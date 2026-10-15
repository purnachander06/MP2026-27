"""Groq LLM explanation provider using OpenAI-compatible HTTP interface via httpx."""

import json
import logging
import os
import time
from typing import Optional, Dict, Any
import httpx

from src.explanations.providers.base import BaseExplanationProvider
from src.explanations.prompt_builder import build_explanation_prompt
from src.explanations.schemas import (
    SeverityLevel,
    ThreatExplanationResponse,
    ThreatPredictionResponse,
)

logger = logging.getLogger("GroqExplanationProvider")

GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
# Supported Groq models from user tier: openai/gpt-oss-20b, qwen/qwen3.8-27b, openai/gpt-oss-120b
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
SUPPORTED_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-safeguard-20b",
]


class GroqExplanationProvider(BaseExplanationProvider):
    """
    Asynchronous LLM explanation provider integrating with Groq's API.
    Uses Llama-3.3-70b-versatile with strict JSON mode, low temperature, and bounded timeouts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: float = 15.0,
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_GROQ_MODEL)
        self.timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", timeout_seconds))
        self.max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", max_tokens))

    @property
    def provider_name(self) -> str:
        return f"groq:{self.model}"

    def is_available(self) -> bool:
        """Available if a non-empty API key is configured."""
        return bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

    async def generate_explanation(
        self,
        prediction: ThreatPredictionResponse,
        operator_context: Optional[str] = None,
    ) -> ThreatExplanationResponse:
        """Call Groq API, validate response, and return ThreatExplanationResponse."""
        if not self.is_available():
            raise RuntimeError("Groq API key not configured. Cannot generate online explanation.")

        system_prompt, user_prompt = build_explanation_prompt(
            prediction_id=prediction.prediction_id,
            predicted_class=prediction.predicted_class,
            detector_name=prediction.detector_name,
            malicious_probability=prediction.malicious_probability,
            confidence_score=prediction.confidence_score,
            validated_evidence=prediction.validated_evidence,
            model_version=prediction.model_version,
            operator_context=operator_context,
        )

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }

        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(GROQ_API_ENDPOINT, headers=headers, json=payload)

            if resp.status_code == 429:
                raise RuntimeError(f"Groq API rate limit reached (HTTP 429): {resp.text}")
            elif resp.status_code == 401:
                raise RuntimeError("Groq API authentication failed (HTTP 401). Invalid API key.")
            elif resp.status_code != 200:
                raise RuntimeError(f"Groq API returned HTTP {resp.status_code}: {resp.text}")

            data = resp.json()
            content_str = data["choices"][0]["message"]["content"]
            parsed_json = json.loads(content_str)

        except httpx.TimeoutException as e:
            raise TimeoutError(f"Groq API request timed out after {self.timeout_seconds}s: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Groq API returned non-JSON content: {e}") from e
        except Exception as e:
            logger.error(f"Groq provider call failed: {e}")
            raise

        # Validate severity
        raw_sev = str(parsed_json.get("severity", "medium")).lower().strip()
        try:
            severity = SeverityLevel(raw_sev)
        except ValueError:
            severity = SeverityLevel.MEDIUM if prediction.predicted_class == "malicious" else SeverityLevel.INFORMATIONAL

        return ThreatExplanationResponse(
            prediction_id=prediction.prediction_id,
            predicted_class=prediction.predicted_class,
            detector_name=prediction.detector_name,
            confidence_score=prediction.confidence_score,
            severity=severity,
            severity_rationale=str(parsed_json.get("severity_rationale", "Generated by Groq LLM reasoning engine.")),
            summary=str(parsed_json.get("summary", "Automated threat explanation generated from flow telemetry.")),
            supporting_evidence=list(parsed_json.get("supporting_evidence", ["Flow telemetry analyzed by Groq model."])),
            uncertainties=list(parsed_json.get("uncertainties", ["Analysis based solely on header metadata."])),
            recommended_actions=list(parsed_json.get("recommended_actions", ["Perform standard network log review."])),
            requires_human_review=bool(parsed_json.get("requires_human_review", prediction.predicted_class == "malicious")),
            explanation_provider=self.provider_name,
            model_version=prediction.model_version,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            disclaimer="Advisory automated explanation grounded in flow telemetry. Not proof of compromise. Low-risk human triage required.",
            is_fallback=False,
        )
