"""
Tests for parsing model replies into allocations.

Two of these encode bugs found during development that produced *plausible
wrong answers* rather than errors — the failure mode that silently corrupts
a response kernel:

  1. Per-ticker proximity search: in "40% in AAPL, 35% in MSFT" the 40% falls
     within the proximity window of MSFT and gets claimed twice.
  2. Greedy nearest-neighbour with a claim set: 35% is nearer to AAPL than to
     MSFT, finds AAPL taken, and is discarded — MSFT ends up with no weight.

Both returned a valid-looking simplex point. Only comparing against a known
answer caught them.
"""

import numpy as np
import pytest

from e5.parsing import ParseStatus, parse_allocation

T = ["AAPL", "MSFT", "NVDA"]


def alloc(text, tickers=None):
    return parse_allocation(text, tickers or T)


class TestJson:
    def test_bare_object(self):
        r = alloc('{"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2}')
        assert r.status is ParseStatus.OK
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])

    def test_fenced_block(self):
        r = alloc('Sure:\n```json\n{"AAPL":0.5,"MSFT":0.3,"NVDA":0.2}\n```')
        assert r.status is ParseStatus.OK

    def test_percent_strings(self):
        r = alloc('{"AAPL":"50%","MSFT":"30%","NVDA":"20%"}')
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])

    def test_dollar_prefix_keys(self):
        r = alloc('{"$AAPL": 0.6, "$MSFT": 0.4, "$NVDA": 0.0}')
        np.testing.assert_allclose(r.allocation, [0.6, 0.4, 0.0])

    def test_last_object_wins(self):
        """Models often reason aloud then restate a final answer."""
        r = alloc('First thought {"AAPL":1.0}. On reflection: '
                  '{"AAPL":0.4,"MSFT":0.4,"NVDA":0.2}')
        np.testing.assert_allclose(r.allocation, [0.4, 0.4, 0.2])

    def test_json_beats_prose(self):
        r = alloc('I would put 90% in AAPL. {"AAPL":0.5,"MSFT":0.3,"NVDA":0.2}')
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])


class TestProse:
    def test_number_before_ticker(self):
        """Regression: the per-ticker proximity search bound 40% to MSFT."""
        r = alloc("Put 40% in AAPL, 35% in MSFT, and 25% in NVDA.")
        assert r.status is ParseStatus.OK
        np.testing.assert_allclose(r.allocation, [0.40, 0.35, 0.25])

    def test_ticker_before_number(self):
        """Regression: greedy nearest-unclaimed dropped MSFT entirely."""
        r = alloc("AAPL: 50%\nMSFT: 30%\nNVDA: 20%")
        assert r.status is ParseStatus.OK
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])

    def test_decimals_without_percent(self):
        r = alloc("AAPL 0.5, MSFT 0.3, NVDA 0.2")
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])

    def test_markdown_table(self):
        r = alloc("| AAPL | 45% |\n| MSFT | 35% |\n| NVDA | 20% |")
        np.testing.assert_allclose(r.allocation, [0.45, 0.35, 0.20])


class TestStatuses:
    def test_renormalizes_and_records_raw_sum(self):
        r = alloc('{"AAPL":0.5,"MSFT":0.3,"NVDA":0.15}')
        assert r.status is ParseStatus.RENORMALIZED
        assert r.raw_sum == pytest.approx(0.95)
        np.testing.assert_allclose(r.allocation.sum(), 1.0)

    def test_partial_when_tickers_omitted(self):
        r = alloc('{"AAPL":1.0}')
        assert r.status is ParseStatus.PARTIAL
        assert r.found == ("AAPL",)

    def test_ambiguous_when_counts_disagree(self):
        """A distractor number (a past return) means prose binding cannot be
        trusted. Flagged rather than silently accepted — telling
        'AAPL returned 30%' from 'AAPL 50%' needs semantics, not regex."""
        r = alloc("Over 12 months AAPL returned 30%. I suggest "
                  "AAPL 50%, MSFT 30%, NVDA 20%.")
        assert r.status is ParseStatus.AMBIGUOUS

    def test_refusal_has_no_weights(self):
        r = alloc("I cannot provide financial advice.")
        assert r.status is ParseStatus.NO_WEIGHTS
        assert not r.usable

    def test_empty(self):
        assert alloc("").status is ParseStatus.EMPTY
        assert alloc("   \n ").status is ParseStatus.EMPTY

    def test_usable_flag(self):
        assert alloc('{"AAPL":0.5,"MSFT":0.3,"NVDA":0.2}').usable
        assert not alloc("no numbers here").usable


class TestInvariants:
    @pytest.mark.parametrize("text", [
        '{"AAPL":0.5,"MSFT":0.3,"NVDA":0.2}',
        '{"AAPL":0.5,"MSFT":0.3,"NVDA":0.15}',
        '{"AAPL":1.0}',
        "Put 40% in AAPL, 35% in MSFT, and 25% in NVDA.",
        "I cannot provide financial advice.",
        "",
        '{"AAPL":-0.5,"MSFT":1.5}',
        "AAPL AAPL AAPL",
    ])
    def test_always_returns_a_simplex_point(self, text):
        """Downstream metrics validate their input, so a malformed vector
        would raise deep in the pipeline instead of being recorded here."""
        r = alloc(text)
        assert r.allocation.shape == (len(T),)
        assert (r.allocation >= 0).all()
        np.testing.assert_allclose(r.allocation.sum(), 1.0, atol=1e-9)

    def test_negative_weights_clipped(self):
        r = alloc('{"AAPL":-0.5,"MSFT":1.0,"NVDA":0.5}')
        assert (r.allocation >= 0).all()

    def test_raw_text_preserved(self):
        text = "some reply"
        assert alloc(text).raw_text == text

    def test_empty_universe_raises(self):
        with pytest.raises(ValueError, match="empty ticker"):
            parse_allocation("anything", [])

    def test_case_insensitive_tickers(self):
        r = alloc('{"aapl":0.5,"msft":0.3,"nvda":0.2}')
        np.testing.assert_allclose(r.allocation, [0.5, 0.3, 0.2])
