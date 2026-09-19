"""
Tests for the E5 runner and providers.

Everything here runs offline. LiveProvider is exercised only for its
key-handling and never over the network — a test that needs a key would
break the repo's zero-key policy.
"""

import json
from datetime import date

import numpy as np
import pytest

from e5.corpus import build_corpus
from e5.providers import (
    LiveProvider,
    Provider,
    ProviderError,
    ReplayProvider,
    StubProvider,
)
from e5.runner import load_records, run_audit, to_panel

U = ["AAPL", "MSFT", "NVDA", "TSLA"]
DATES = [date(2026, 8, 1)]


@pytest.fixture
def items():
    return build_corpus(U, DATES, max_ticker_groups=2)


@pytest.fixture
def run(tmp_path, items):
    provs = [StubProvider("a"), StubProvider("b", bias="AAPL", bias_weight=0.6)]
    return run_audit(items, provs, U, tmp_path / "run.jsonl", repeats=2), items, provs


class TestProviders:
    def test_stub_satisfies_protocol(self):
        assert isinstance(StubProvider(), Provider)

    def test_stub_emits_parseable_json(self, items):
        """The stub goes through the same parser a live reply does, so stub
        runs still catch parser regressions."""
        raw = StubProvider().complete(items[0].prompt, query=items[0].query)
        obj = json.loads(raw)
        assert set(obj) == set(items[0].tickers)
        assert sum(obj.values()) == pytest.approx(1.0, abs=1e-5)

    def test_stub_is_deterministic(self, items):
        p = StubProvider()
        assert p.complete(items[0].prompt, query=items[0].query) == p.complete(
            items[0].prompt, query=items[0].query
        )

    def test_bias_tilts_allocation(self, items):
        base = json.loads(StubProvider().complete(items[0].prompt, query=items[0].query))
        tilt = json.loads(
            StubProvider(bias="AAPL", bias_weight=0.8).complete(
                items[0].prompt, query=items[0].query
            )
        )
        assert tilt["AAPL"] > base["AAPL"]

    def test_stub_requires_the_query(self):
        with pytest.raises(ProviderError, match="RetailQuery"):
            StubProvider().complete("a bare prompt")

    def test_live_provider_never_calls_without_a_key(self, monkeypatch):
        monkeypatch.delenv("E5_TEST_KEY", raising=False)
        p = LiveProvider("x", "some-model", "E5_TEST_KEY")
        with pytest.raises(ProviderError, match="E5_TEST_KEY is unset"):
            p.complete("hello")

    def test_live_provider_rejects_placeholder_key(self, monkeypatch):
        monkeypatch.setenv("E5_TEST_KEY", "[YOUR_KEY_HERE]")
        with pytest.raises(ProviderError, match="unset"):
            LiveProvider("x", "m", "E5_TEST_KEY").complete("hello")


class TestReplay:
    def test_round_trips_a_recording(self, run):
        path, items, _ = run
        rp = ReplayProvider(path, "a")
        assert len(rp) > 0
        rp._qid_hint = items[0].query_id
        assert json.loads(rp.complete(items[0].prompt, query=items[0].query))

    def test_missing_key_raises_rather_than_defaulting(self, run):
        """A silent fallback would make a partial recording look complete."""
        path, _, _ = run
        rp = ReplayProvider(path, "a")
        rp._qid_hint = "nonexistent"
        with pytest.raises(ProviderError, match="no recorded reply"):
            rp.complete("p")

    def test_unknown_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReplayProvider(tmp_path / "nope.jsonl", "a")


class TestRunAudit:
    def test_record_count(self, run):
        path, items, provs = run
        assert len(load_records(path)) == len(items) * len(provs) * 2

    def test_allocations_padded_to_universe(self, run):
        """Items ask about ticker subsets; metrics need a common basis."""
        for r in load_records(run[0]):
            assert len(r.allocation) == len(U)
            assert sum(r.allocation) == pytest.approx(1.0, abs=1e-6)

    def test_resume_skips_completed_work(self, run):
        path, items, provs = run
        before = len(load_records(path))
        run_audit(items, provs, U, path, repeats=2)
        assert len(load_records(path)) == before

    def test_provider_failure_is_recorded_not_fatal(self, tmp_path, items):
        class Broken:
            name = "broken"

            def complete(self, prompt, *, query=None):
                raise ProviderError("simulated outage")

        path = run_audit(items[:3], [Broken(), StubProvider("ok")], U,
                         tmp_path / "r.jsonl")
        recs = load_records(path)
        assert any(r.error for r in recs), "failure not recorded"
        assert any(r.error is None for r in recs), "run aborted on failure"

    def test_raw_text_is_preserved(self, run):
        assert all(r.raw_text for r in load_records(run[0]) if not r.error)


class TestToPanel:
    def test_shape(self, run):
        recs = load_records(run[0])
        one = [r for r in recs if r.query_id == recs[0].query_id]
        panel, names = to_panel(one)
        assert panel.shape == (2, 2, len(U))
        assert names == ["a", "b"]

    def test_panel_rows_are_simplex_points(self, run):
        recs = load_records(run[0])
        panel, _ = to_panel([r for r in recs if r.query_id == recs[0].query_id])
        np.testing.assert_allclose(panel.sum(axis=-1), 1.0, atol=1e-6)

    def test_status_filter_is_a_parameter(self, run):
        """Excluding sloppy replies is an analysis choice, not baked in."""
        recs = load_records(run[0])
        one = [r for r in recs if r.query_id == recs[0].query_id]
        with pytest.raises(ValueError, match="no records"):
            to_panel(one, include_statuses=("empty",))

    def test_feeds_metrics(self, run):
        from e5.metrics import crowding_index, spread

        recs = load_records(run[0])
        panel, _ = to_panel([r for r in recs if r.query_id == recs[0].query_id])
        assert 0.0 <= crowding_index(panel) <= 1.0
        assert spread(panel) >= 0.0
