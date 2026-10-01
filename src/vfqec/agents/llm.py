"""Small provider-neutral JSON advisor with a deterministic no-key fallback."""

import json
import os

import httpx


class LLM:
    """LLMs advise orchestration only, never receive plant fields or optimizer observations."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "none").lower()
        self.key = os.getenv("LLM_API_KEY", "")

    def advise(self, role: str, context: dict, fallback: dict) -> dict:
        if self.provider == "none" or not self.key:
            return dict(fallback, advisor="rule-based")
        system = (
            "You advise a quantum simulation experiment. Return one JSON object only. "
            "Do not invent results, request secrets, or propose shell commands. "
            "Advice never changes physical results. Role: " + role
        )
        prompt = json.dumps({"context": context, "response_shape": fallback})
        model = os.getenv("LLM_MODEL")
        try:
            with httpx.Client(timeout=30) as client:
                if self.provider == "anthropic":
                    if not model:
                        raise ValueError("Set LLM_MODEL for Anthropic")
                    response = client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                        json={
                            "model": model,
                            "max_tokens": 800,
                            "system": system,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    response.raise_for_status()
                    answer = response.json()["content"][0]["text"]
                elif self.provider in ("openai", "openai-compatible"):
                    if not model:
                        raise ValueError("Set LLM_MODEL for the selected provider")
                    base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                    response = client.post(
                        base + "/chat/completions",
                        headers={"Authorization": "Bearer " + self.key},
                        json={
                            "model": model,
                            "max_tokens": 800,
                            "response_format": {"type": "json_object"},
                            "messages": [
                                {"role": "system", "content": system},
                                {"role": "user", "content": prompt},
                            ],
                        },
                    )
                    response.raise_for_status()
                    answer = response.json()["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Unknown LLM_PROVIDER")
            parsed = json.loads(answer)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object")
            return dict(parsed, advisor=self.provider)
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            # Do not record HTTP bodies or request headers, which might contain credentials.
            return dict(fallback, advisor="rule-based (LLM unavailable or invalid response)")
