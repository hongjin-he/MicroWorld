"""
Parse a model's free-text reply into an allocation over the ticker universe.

Design note: this parses *tolerantly* and records what it found, rather than
accepting only well-formed output. If model A emits clean JSON 99% of the
time and model B 80%, discarding failures compares A's full response
distribution against B's tidiest 80% — which understates B's dispersion,
the very quantity E5 measures. Format compliance plausibly correlates with
templated advice, so silent rejection would bias h upward.

Every parse therefore returns a ParseResult carrying the raw text, what was
extracted, the pre-normalization sum, and a status. The decision to exclude
anything is made at analysis time, not here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

__all__ = ["ParseStatus", "ParseResult", "parse_allocation"]


class ParseStatus(str, Enum):
    OK = "ok"                    # weights found, summed to ~1
    RENORMALIZED = "renormalized"  # weights found, sum was off by > tol
    PARTIAL = "partial"          # some tickers found, others missing
    AMBIGUOUS = "ambiguous"      # prose fallback, number/ticker counts disagreed
    NO_WEIGHTS = "no_weights"    # nothing numeric attributable to a ticker
    EMPTY = "empty"              # blank or whitespace reply


@dataclass(frozen=True)
class ParseResult:
    status: ParseStatus
    allocation: NDArray[np.float64]  # always a valid simplex point
    raw_sum: float                   # sum before normalization
    found: tuple[str, ...]           # tickers an explicit weight was found for
    raw_text: str

    @property
    def usable(self) -> bool:
        """True when a weight was attributable to at least one ticker."""
        return self.status in (
            ParseStatus.OK,
            ParseStatus.RENORMALIZED,
            ParseStatus.PARTIAL,
        )


_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _num(x) -> float | None:
    """Coerce a JSON value or string to a fraction. '25%' -> 0.25."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        s = x.strip().rstrip("%")
        try:
            v = float(s)
        except ValueError:
            return None
        return v / 100.0 if x.strip().endswith("%") else v
    return None


def _from_json(text: str, tickers: list[str]) -> dict[str, float]:
    """Try fenced blocks first, then any bare {...}. Last valid one wins,
    since models often restate a final answer after reasoning aloud."""
    out: dict[str, float] = {}
    upper = {t.upper(): t for t in tickers}
    candidates = _FENCE.findall(text) + _JSON_BLOCK.findall(text)
    for blob in candidates:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        found = {}
        for k, v in obj.items():
            key = str(k).strip().upper().lstrip("$")
            if key in upper and (n := _num(v)) is not None:
                found[upper[key]] = n
        if found:
            out = found
    return out


def _from_text(text: str, tickers: list[str]) -> tuple[dict[str, float], bool]:
    """
    Fallback for prose: 'AAPL: 40%', '40% in AAPL', tables, comma lists.

    Pairs the k-th number with the k-th ticker mention when the counts match,
    which is how models actually write these lists. Two rejected approaches:
    a per-ticker proximity search mis-binds on lists ("40% in AAPL, 35% in
    MSFT" puts 40% within 20 chars of MSFT, claiming it twice), and greedy
    nearest-neighbour with a claim set silently drops numbers whose nearest
    ticker was already taken. Falls back to nearest-unclaimed only when the
    counts differ.
    """
    seen: list[tuple[int, str]] = []
    for t in tickers:
        for m in re.finditer(rf"\${{0,1}}\b{re.escape(t)}\b", text, re.IGNORECASE):
            seen.append((m.start(), t))
    if not seen:
        return {}, False
    seen.sort()
    # first mention of each ticker, in order of appearance
    order: list[str] = []
    for _, t in seen:
        if t not in order:
            order.append(t)

    nums: list[tuple[int, float]] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(%)|(?<![\d.])(0?\.\d+)", text):
        raw = m.group(1) or m.group(3)
        val = float(raw) / 100.0 if m.group(2) else float(raw)
        nums.append((m.start(), val))
    if not nums:
        return {}, False

    if len(nums) == len(order):
        return {t: v for t, (_, v) in zip(order, nums)}, False

    # counts disagree: distractor numbers (past returns, dates) are in play.
    # Distinguishing "AAPL returned 30%" from "AAPL 50%" needs semantics,
    # not pattern matching, so flag rather than pretend.
    ambiguous = True

    out: dict[str, float] = {}
    for pos, val in nums:
        cands = [(abs(p - pos), t) for p, t in seen if t not in out]
        if not cands:
            break
        d, t = min(cands)
        if d <= 40:
            out[t] = val
    return out, ambiguous


def parse_allocation(
    text: str,
    tickers: list[str],
    sum_tol: float = 0.02,
) -> ParseResult:
    """
    Extract an allocation over `tickers`. Always returns a valid simplex
    point so downstream metrics never see a malformed vector; inspect
    `status` and `raw_sum` to decide whether to include it.

    Unmentioned tickers get weight 0 when at least one weight was found.
    A reply with no attributable weights falls back to equal weight with
    status NO_WEIGHTS — recorded, not silently treated as a real answer.
    """
    n = len(tickers)
    if n == 0:
        raise ValueError("empty ticker universe")
    uniform = np.full(n, 1.0 / n)

    if not text or not text.strip():
        return ParseResult(ParseStatus.EMPTY, uniform, 0.0, (), text or "")

    weights = _from_json(text, tickers)
    from_prose = False
    if not weights:
        weights, prose_ambiguous = _from_text(text, tickers)
        from_prose = True
    if not weights:
        return ParseResult(ParseStatus.NO_WEIGHTS, uniform, 0.0, (), text)

    vec = np.array([max(0.0, weights.get(t, 0.0)) for t in tickers], dtype=float)
    raw_sum = float(vec.sum())
    if raw_sum <= 0:
        return ParseResult(ParseStatus.NO_WEIGHTS, uniform, raw_sum, (), text)

    alloc = vec / raw_sum
    found = tuple(t for t in tickers if t in weights)

    if from_prose and prose_ambiguous:
        status = ParseStatus.AMBIGUOUS
    elif len(found) < n and abs(raw_sum - 1.0) <= sum_tol:
        status = ParseStatus.PARTIAL
    elif abs(raw_sum - 1.0) > sum_tol:
        status = ParseStatus.RENORMALIZED
    else:
        status = ParseStatus.OK

    return ParseResult(status, alloc, raw_sum, found, text)
