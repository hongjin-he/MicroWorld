"""
E5 runner: corpus -> providers -> recorded responses.

Records to append-only JSONL so an interrupted run resumes rather than
restarts, and a partial run is still analyzable. Every response is stored
with its raw text, parse status, and pre-normalization sum; nothing is
discarded at write time (see e5.parsing for why).

Allocations are padded to the full ticker universe. A corpus item asks about
a subset, but e5.metrics needs every response on a common basis to stack
into a (models, repeats, assets) panel — unpadded vectors of differing
length cannot be compared.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from e5.corpus import CorpusItem
from e5.parsing import ParseStatus, parse_allocation
from e5.providers import Provider, ProviderError

__all__ = ["ResponseRecord", "run_audit", "load_records", "to_panel"]


@dataclass(frozen=True)
class ResponseRecord:
    query_id: str
    model: str
    repeat: int
    as_of: str
    archetype: str
    condition: str
    tickers: list[str]
    universe: list[str]
    prompt: str
    raw_text: str
    status: str
    raw_sum: float
    allocation: list[float]   # padded to `universe`
    error: str | None = None
    recorded_at: str = ""


def _pad(alloc: NDArray[np.float64], subset: Sequence[str], universe: Sequence[str]) -> list[float]:
    """Place a subset allocation into the full-universe basis."""
    idx = {t: i for i, t in enumerate(universe)}
    out = np.zeros(len(universe))
    for t, w in zip(subset, alloc):
        if t in idx:
            out[idx[t]] = w
    s = out.sum()
    return (out / s if s > 0 else np.full(len(universe), 1.0 / len(universe))).tolist()


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    """(query_id, model, repeat) already recorded, so a resumed run skips them."""
    if not path.exists():
        return set()
    seen = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add((r["query_id"], r["model"], r.get("repeat", 0)))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from an interrupted run
    return seen


def run_audit(
    items: Iterable[CorpusItem],
    providers: Sequence[Provider],
    universe: Sequence[str],
    out_path: str | Path,
    repeats: int = 1,
    resume: bool = True,
    progress: bool = False,
) -> Path:
    """
    Send every (item x provider x repeat) and append one JSON line each.

    Provider failures are recorded with `error` set and an equal-weight
    placeholder, then the run continues — losing an entire audit to one
    timeout would be worse than a recorded gap that analysis can exclude.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    items = list(items)
    done = _done_keys(out) if resume else set()
    n_written = 0

    with out.open("a") as fh:
        for item in items:
            for provider in providers:
                for rep in range(repeats):
                    if (item.query_id, provider.name, rep) in done:
                        continue
                    # ReplayProvider keys off query_id, which RetailQuery lacks
                    if hasattr(provider, "_qid_hint"):
                        provider._qid_hint = item.query_id

                    error = None
                    try:
                        raw = provider.complete(item.prompt, query=item.query)
                    except ProviderError as exc:
                        raw, error = "", str(exc)

                    parsed = parse_allocation(raw, list(item.tickers))
                    rec = ResponseRecord(
                        query_id=item.query_id,
                        model=provider.name,
                        repeat=rep,
                        as_of=item.as_of.isoformat(),
                        archetype=item.archetype.value,
                        condition=item.condition.name,
                        tickers=list(item.tickers),
                        universe=list(universe),
                        prompt=item.prompt,
                        raw_text=parsed.raw_text,
                        status=parsed.status.value,
                        raw_sum=parsed.raw_sum,
                        allocation=_pad(parsed.allocation, item.tickers, universe),
                        error=error,
                        recorded_at=datetime.now(timezone.utc).isoformat(),
                    )
                    fh.write(json.dumps(asdict(rec)) + "\n")
                    fh.flush()  # a crash loses at most the current call
                    n_written += 1
                    if progress and n_written % 25 == 0:
                        print(f"  {n_written} responses", flush=True)
    return out


def load_records(path: str | Path) -> list[ResponseRecord]:
    """Read a run back, skipping any torn final line."""
    out = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ResponseRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def to_panel(
    records: Sequence[ResponseRecord],
    models: Sequence[str] | None = None,
    include_statuses: Sequence[str] = (
        ParseStatus.OK.value,
        ParseStatus.RENORMALIZED.value,
        ParseStatus.PARTIAL.value,
    ),
) -> tuple[NDArray[np.float64], list[str]]:
    """
    Stack records into the (n_models, n_repeats, n_assets) panel e5.metrics
    expects, for one query.

    `include_statuses` is a parameter rather than a fixed policy: excluding
    unparseable replies compares each model's tidiest subset, which biases
    dispersion downward for models that follow instructions less reliably.
    The default keeps everything with an attributable weight.
    """
    usable = [r for r in records if r.status in include_statuses]
    if not usable:
        raise ValueError("no records with the requested statuses")
    names = list(models) if models else sorted({r.model for r in usable})
    by_model = {m: [r.allocation for r in usable if r.model == m] for m in names}
    missing = [m for m, v in by_model.items() if not v]
    if missing:
        raise ValueError(f"no usable records for {missing}")
    depth = min(len(v) for v in by_model.values())
    panel = np.array([by_model[m][:depth] for m in names], dtype=float)
    return panel, names
