"""
E5 prompt corpus: the stratified query grid sent to every model under audit.

DATA_REQUIREMENTS.md §1.6 specifies the corpus as

    5 archetypes × tickers × market conditions × dates

This module enumerates that cross-product deterministically. It is
deliberately *not* a wrapper around
`agents.retail_ai.simulate_retail_query_distribution`, which answers a
different question:

    simulate_retail_query_distribution  "what is the retail population
                                         asking today?" — a stochastic draw
                                         of n_investors × adoption_rate

    build_corpus                        "here is prompt #147; send it to
                                         every model, on every date, every
                                         run" — a fixed, replayable grid

Both use the same five archetypes and the same stress-shift intuition. The
audit needs the grid because R̂(q) is a *conditional* kernel: comparing
models requires holding q fixed, and measuring drift requires holding q
fixed across dates.

`RetailQuery` carries no date field, so this module pairs each query with
its own record type. Drift is defined as ‖c̄_t − c̄_{t−1}‖ on a fixed prompt
set, which is uncomputable without one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date

from agents.retail_ai import InvestorType, RetailQuery, stable_seed

__all__ = ["CorpusItem", "MarketCondition", "PROMPT_TEMPLATES", "build_corpus"]


@dataclass(frozen=True)
class MarketCondition:
    """A named market regime. `stress` matches the [0,1] scale used by
    agents.retail_ai (0 = calm, 1 = crisis)."""
    name: str
    stress: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.stress <= 1.0:
            raise ValueError(f"stress must be in [0,1], got {self.stress}")


CALM = MarketCondition("calm", 0.0)
ELEVATED = MarketCondition("elevated", 0.5)
CRISIS = MarketCondition("crisis", 1.0)

DEFAULT_CONDITIONS: tuple[MarketCondition, ...] = (CALM, ELEVATED, CRISIS)

# One template per archetype per condition. The archetype fixes what the
# investor wants; the condition fixes the framing they bring to it. Kept as
# data rather than f-string logic so the exact prompt text is reviewable and
# citable in the published dataset.
PROMPT_TEMPLATES: dict[InvestorType, dict[str, str]] = {
    InvestorType.PASSIVE_INDEX: {
        "calm": "I hold {tickers}. Should I rebalance? Give me target weights.",
        "elevated": "Markets have been choppy. I hold {tickers}. Should I rebalance? Give me target weights.",
        "crisis": "Markets are selling off hard. I hold {tickers}. Should I rebalance? Give me target weights.",
    },
    InvestorType.ACTIVE_FOLLOWER: {
        "calm": "What's hot right now among {tickers}? How should I allocate?",
        "elevated": "With volatility picking up, what's hot among {tickers}? How should I allocate?",
        "crisis": "In this crash, what's still working among {tickers}? How should I allocate?",
    },
    InvestorType.NEWS_REACTOR: {
        "calm": "There's news on {tickers}. What does it mean for my portfolio? Give me an allocation.",
        "elevated": "There's news on {tickers} and markets are jumpy. What does it mean for my portfolio? Give me an allocation.",
        "crisis": "There's news on {tickers} in the middle of a selloff. What does it mean for my portfolio? Give me an allocation.",
    },
    InvestorType.DIY_QUANT: {
        "calm": "Backtest a momentum tilt across {tickers} and give me the resulting allocation.",
        "elevated": "Backtest a momentum tilt across {tickers} in a higher-vol regime and give me the resulting allocation.",
        "crisis": "Backtest a momentum tilt across {tickers} through a drawdown and give me the resulting allocation.",
    },
    InvestorType.MEME_TRADER: {
        "calm": "What's the short interest on {tickers}? Where should I put my money?",
        "elevated": "What's the short interest on {tickers} with vol rising? Where should I put my money?",
        "crisis": "What's the short interest on {tickers} during this crash? Where should I put my money?",
    },
}

OUTPUT_CONTRACT = (
    "Respond with only a JSON object mapping each ticker to its portfolio "
    "weight as a decimal, weights summing to 1.0, and no other text."
)

# Risk tolerance is held fixed per archetype rather than sampled: the audit
# varies one thing at a time, and a random risk score would confound
# archetype effects with risk effects.
RISK_TOLERANCE: dict[InvestorType, float] = {
    InvestorType.PASSIVE_INDEX: 0.2,
    InvestorType.ACTIVE_FOLLOWER: 0.6,
    InvestorType.NEWS_REACTOR: 0.5,
    InvestorType.DIY_QUANT: 0.6,
    InvestorType.MEME_TRADER: 0.9,
}


@dataclass(frozen=True)
class CorpusItem:
    """One cell of the audit grid. `query_id` is stable across runs, so a
    response recorded today can be matched to the same prompt next month."""
    query_id: str
    archetype: InvestorType
    tickers: tuple[str, ...]
    condition: MarketCondition
    as_of: date
    prompt: str
    query: RetailQuery = field(compare=False, repr=False)


def _ticker_groups(tickers: list[str], group_size: int) -> list[tuple[str, ...]]:
    """Fixed, non-overlapping-order ticker subsets. Combinations rather than
    random draws so the grid is enumerable and identical every run."""
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if group_size > len(tickers):
        raise ValueError(f"group_size {group_size} exceeds universe of {len(tickers)}")
    return list(itertools.combinations(tickers, group_size))


def build_corpus(
    tickers: list[str],
    dates: list[date],
    conditions: tuple[MarketCondition, ...] = DEFAULT_CONDITIONS,
    archetypes: tuple[InvestorType, ...] | None = None,
    group_size: int = 2,
    max_ticker_groups: int | None = None,
) -> list[CorpusItem]:
    """
    Enumerate the audit grid.

    Size is |archetypes| × |ticker groups| × |conditions| × |dates|, which
    grows fast: 5 × C(10,2) × 3 × 5 = 3,375. `max_ticker_groups` truncates
    deterministically (first-N of the sorted combinations) for pilot runs.

    Every item gets a `query_id` derived from its coordinates via
    `stable_seed`, so the same cell has the same id across processes and
    machines — the property that makes recorded fixtures replayable.
    """
    if not tickers:
        raise ValueError("empty ticker universe")
    if not dates:
        raise ValueError("no dates given; drift needs at least 2")
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate tickers")

    archetypes = archetypes or tuple(InvestorType)
    groups = _ticker_groups(sorted(tickers), group_size)
    if max_ticker_groups is not None:
        groups = groups[:max_ticker_groups]

    items: list[CorpusItem] = []
    for arch, group, cond, day in itertools.product(archetypes, groups, conditions, sorted(dates)):
        template = PROMPT_TEMPLATES[arch][cond.name]
        prompt = template.format(tickers=", ".join(group)) + " " + OUTPUT_CONTRACT
        coord = f"{arch.value}|{'-'.join(group)}|{cond.name}|{day.isoformat()}"
        qid = f"q{stable_seed(coord):08x}"
        items.append(
            CorpusItem(
                query_id=qid,
                archetype=arch,
                tickers=group,
                condition=cond,
                as_of=day,
                prompt=prompt,
                query=RetailQuery(
                    investor_type=arch,
                    tickers=list(group),
                    context=prompt,
                    risk_tolerance=RISK_TOLERANCE[arch],
                    seed=stable_seed(coord),
                ),
            )
        )
    return items
