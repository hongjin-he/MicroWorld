"""
Providers for the E5 audit.

The zero-key policy in the README ("every demo and test runs fully offline")
means live calls must be the exception, not the default. Three
implementations sit behind one structural protocol:

    StubProvider    wraps agents.retail_ai.stub_llm_response — a
                    zero-variance baseline needing no network
    ReplayProvider  reads a recorded JSONL run, so one real audit becomes a
                    permanent offline fixture
    LiveProvider    the only one that touches the network or needs a key

Because Provider is a Protocol rather than a base class, tests need no
mocking library: any object with .name and .complete() qualifies.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from agents.retail_ai import RetailQuery, stub_llm_response

__all__ = ["Provider", "StubProvider", "ReplayProvider", "LiveProvider", "ProviderError"]


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a reply. Recorded, not fatal."""


@runtime_checkable
class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, query: RetailQuery | None = None) -> str:
        """Return the model's raw reply text."""
        ...


class StubProvider:
    """
    Deterministic baseline. Emits JSON so it exercises the same parsing path
    a live reply would, rather than bypassing it — otherwise stub runs would
    not catch parser regressions.

    `bias` optionally tilts allocations toward one ticker, so multi-model
    scenarios with known ground-truth agreement can be constructed for tests.
    """

    def __init__(self, name: str = "stub", bias: str | None = None, bias_weight: float = 0.0):
        self.name = name
        self.bias = bias
        self.bias_weight = float(np.clip(bias_weight, 0.0, 1.0))

    def complete(self, prompt: str, *, query: RetailQuery | None = None) -> str:
        if query is None:
            raise ProviderError("StubProvider needs the RetailQuery, not just the prompt")
        tickers = list(query.tickers)
        alloc = stub_llm_response(query, tickers)
        if self.bias and self.bias in tickers and self.bias_weight > 0:
            target = np.zeros(len(tickers))
            target[tickers.index(self.bias)] = 1.0
            alloc = (1 - self.bias_weight) * alloc + self.bias_weight * target
            alloc = alloc / alloc.sum()
        return json.dumps({t: round(float(w), 6) for t, w in zip(tickers, alloc)})


class ReplayProvider:
    """
    Replays a recorded run. Keyed by (model, query_id) so a multi-model
    recording replays each model faithfully.

    Missing keys raise rather than falling back to a default: a silent
    substitution would make a partial recording look like a complete one.
    """

    def __init__(self, path: str | Path, name: str):
        self.name = name
        self._by_qid: dict[str, str] = {}
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no recording at {p}")
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("model") == name:
                    self._by_qid[rec["query_id"]] = rec.get("raw_text", "")

    def __len__(self) -> int:
        return len(self._by_qid)

    def complete(self, prompt: str, *, query: RetailQuery | None = None) -> str:
        qid = getattr(query, "query_id", None) or self._qid_hint
        if qid not in self._by_qid:
            raise ProviderError(f"{self.name}: no recorded reply for {qid!r}")
        return self._by_qid[qid]

    # set by the runner immediately before each call
    _qid_hint: str = ""


class LiveProvider:
    """
    Real API calls. The only provider that needs a key, and the only one
    excluded from the offline test path.

    Reads its key from the environment; never accepts one as a literal, and
    never logs it. Retries transient failures with exponential backoff and
    surfaces the rest as ProviderError so the runner can record the failure
    and continue rather than losing a partial run.
    """

    def __init__(
        self,
        name: str,
        model: str,
        env_var: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        temperature: float = 1.0,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        self.name = name
        self.model = model
        self.env_var = env_var
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def _key(self) -> str:
        key = os.getenv(self.env_var, "")
        if not key or key.startswith("["):
            raise ProviderError(
                f"{self.env_var} is unset. Add it to .env — see .env.example."
            )
        return key

    def complete(self, prompt: str, *, query: RetailQuery | None = None) -> str:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                self.base_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._key()}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read())
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise ProviderError(f"{self.name}: {type(last).__name__} after {self.max_retries} tries")
