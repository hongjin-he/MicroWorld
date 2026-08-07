"""
Tests for groupoid algebra of financial events (events/operators.py).
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from events.operators import (
    EventMode, EventOperator,
    stock_split_operator, dividend_operator, earnings_shock_operator,
    analyst_rating_operator, secondary_offering_operator, share_buyback_operator,
    trading_halt_operator, short_squeeze_operator, index_change_operator,
    rate_change_operator, qe_operator, qt_operator, systemic_crisis_operator,
    circuit_breaker_operator, volatility_regime_shift_operator, inflation_shock_operator,
    merger_operator, spinoff_operator, ipo_operator, delisting_operator, bankruptcy_operator,
    compose, event_sequence,
    P, V, L, K, I,
)


class TestEventModes:
    def test_stock_split_mode_local(self):
        op = stock_split_operator(ratio=2.0, n=1)
        assert op.mode == EventMode.LOCAL

    def test_stock_split_log_price_decreases(self):
        """2:1 split halves share price → log_price shift = -log(2)."""
        op = stock_split_operator(ratio=2.0, n=1)
        np.testing.assert_allclose(op.b_w[P], -np.log(2.0), rtol=1e-6)

    def test_stock_split_shares_outstanding_increases(self):
        """2:1 split doubles shares outstanding → log(shares) shift = +log(2)."""
        op = stock_split_operator(ratio=2.0, n=1)
        np.testing.assert_allclose(op.b_w[K], np.log(2.0), rtol=1e-6)

    def test_dividend_mode_local(self):
        op = dividend_operator(div_yield=0.02, n=1)
        assert op.mode == EventMode.LOCAL

    def test_dividend_lowers_price(self):
        """Dividend of q → price drops by -log(1+q) (ex-date adjustment)."""
        op = dividend_operator(div_yield=0.02, n=1)
        np.testing.assert_allclose(op.b_w[P], -np.log(1.02), rtol=1e-5)

    def test_rate_change_mode_global(self):
        op = rate_change_operator(change_bps=25, n=4)
        assert op.mode == EventMode.GLOBAL

    def test_rate_hike_lowers_prices(self):
        """Rate hike (positive change_bps) should lower all log prices."""
        n = 4
        op = rate_change_operator(change_bps=25, n=n)
        for i in range(n):
            assert op.b_w[i * 5 + P] < 0, f"Asset {i} price should decrease after rate hike"

    def test_rate_cut_raises_prices(self):
        """Rate cut (negative change_bps) should raise all log prices."""
        n = 3
        op = rate_change_operator(change_bps=-50, n=n)
        for i in range(n):
            assert op.b_w[i * 5 + P] > 0, f"Asset {i} price should increase after rate cut"

    def test_merger_mode_pairwise(self):
        op = merger_operator(acquirer_idx=0, target_idx=1, n=2)
        assert op.mode == EventMode.PAIRWISE

    def test_merger_reduces_universe(self):
        """Merger of 2 assets → 1 combined. A_w maps 2d → d (n-1=1)."""
        d = 5
        op = merger_operator(acquirer_idx=0, target_idx=1, n=2)
        assert op.A_w.shape == (d, 2 * d)
        assert op.target_dim == 1
        assert op.source_dim == 2

    def test_ipo_mode_pairwise(self):
        op = ipo_operator(ticker='NEWCO', ipo_price=60.0, n=4)
        assert op.mode == EventMode.PAIRWISE

    def test_ipo_increases_universe(self):
        """IPO: n assets → n+1 assets (one new listing added)."""
        n = 4
        op = ipo_operator(ticker='NEWCO', ipo_price=60.0, n=n)
        assert op.target_dim == n + 1
        assert op.source_dim == n

    def test_ipo_sets_log_price(self):
        """IPO new asset gets log(ipo_price) as its price coordinate."""
        ipo_price = 90.0
        op = ipo_operator(ticker='NEWCO', ipo_price=ipo_price, n=2)
        # New asset state is at the end: indices [2*d : 3*d]
        d = 5
        new_asset_start = 2 * d
        np.testing.assert_allclose(op.b_w[new_asset_start + P], np.log(ipo_price), rtol=1e-6)

    def test_systemic_crisis_global(self):
        op = systemic_crisis_operator(severity=0.8, n=3)
        assert op.mode == EventMode.GLOBAL

    def test_short_squeeze_nonidentity_A(self):
        """Short squeeze is the only Mode I op with A_w ≠ I (momentum self-reinforcement)."""
        op = short_squeeze_operator(squeeze_intensity=0.5, n=1)
        assert op.mode == EventMode.LOCAL
        assert op.A_w[P, P] > 1.0, "Short squeeze must have A_pp > 1 (positive feedback)"


class TestApplyOperator:
    def test_apply_stock_split_shape(self):
        op = stock_split_operator(ratio=2.0, n=1)
        s = np.random.randn(5)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (5,)

    def test_apply_deterministic_with_zero_sigma(self):
        """With Sigma=0, operator is fully deterministic."""
        op = stock_split_operator(ratio=2.0, n=1)
        op.Sigma_w = np.zeros((5, 5))
        s = np.ones(5)
        r1 = op.apply(s, rng=np.random.default_rng(1))
        r2 = op.apply(s, rng=np.random.default_rng(99))
        np.testing.assert_allclose(r1, r2)

    def test_apply_rate_change_full_universe(self):
        """Rate hike acts on full n×d state vector and lowers price components."""
        n, d = 3, 5
        op = rate_change_operator(change_bps=50, n=n)
        s = np.zeros(n * d)
        result = op.apply(s, rng=np.random.default_rng(42))
        assert result.shape == (n * d,)
        for i in range(n):
            assert result[i * d + P] < 0, f"Asset {i} log price should decrease after rate hike"

    def test_merger_reduces_state_size(self):
        """Merger of 2-asset universe produces 1-asset state."""
        n = 2
        op = merger_operator(acquirer_idx=0, target_idx=1, n=n)
        s = np.zeros(n * 5)
        s[P] = np.log(100)    # acquirer at $100
        s[5 + P] = np.log(50) # target at $50
        result = op.apply(s, rng=np.random.default_rng(7))
        assert result.shape == (5,)  # 1 combined entity

    def test_ipo_increases_state_size(self):
        """IPO adds one asset to state vector."""
        n = 3
        op = ipo_operator(ticker='X', ipo_price=50.0, n=n)
        s = np.zeros(n * 5)
        result = op.apply(s, rng=np.random.default_rng(13))
        assert result.shape == ((n + 1) * 5,)


class TestGroupoidComposition:
    def test_mode_i_composes_with_mode_i(self):
        """Two Mode I operators with same dimension compose cleanly."""
        op1 = stock_split_operator(ratio=2.0, n=1)
        op2 = stock_split_operator(ratio=3.0, n=1)
        comp = compose(op1, op2)
        assert comp is not None
        # For identity A: b_comp = b2 + b1
        np.testing.assert_allclose(comp.b_w[P], op1.b_w[P] + op2.b_w[P], rtol=1e-6)

    def test_mode_i_composes_with_mode_ii(self):
        """Mode I (n=3) composes with Mode II (n=3) — same state space."""
        op_macro = rate_change_operator(change_bps=25, n=3)
        op_local = earnings_shock_operator(surprise_pct=10.0, asset_idx=0, n=3)
        comp = compose(op_local, op_macro)   # macro first, then local
        assert comp.source_dim == 3
        assert comp.target_dim == 3

    def test_composition_dimension_mismatch_raises(self):
        """Composing operators with incompatible dimensions must raise ValueError."""
        op_merger = merger_operator(acquirer_idx=0, target_idx=1, n=2)  # 2→1 asset
        with pytest.raises(ValueError):
            compose(op_merger, op_merger)  # output (1 asset) can't feed merger (needs 2)

    def test_ipo_then_split_composition(self):
        """After IPO: n+1 asset universe. Split on that n+1 universe should compose cleanly."""
        n_init = 2
        op_ipo = ipo_operator(ticker='NEW', ipo_price=100.0, n=n_init)   # 2→3 assets
        op_split = stock_split_operator(ratio=2.0, asset_idx=0, n=3)     # 3→3 assets
        comp = compose(op_split, op_ipo)   # IPO then split
        assert comp.source_dim == n_init
        assert comp.target_dim == 3

    def test_composed_operator_inherits_source_target(self):
        """Composed operator takes source from inner (op2), target from outer (op1)."""
        op1 = stock_split_operator(ratio=2.0, n=2)
        op1.target_tickers = ['A', 'B']
        op1.source_tickers = ['A', 'B']
        op2 = dividend_operator(div_yield=0.02, asset_idx=0, n=2)
        op2.source_tickers = ['A', 'B']
        op2.target_tickers = ['A', 'B']
        comp = compose(op1, op2)
        assert comp.source_tickers == op2.source_tickers
        assert comp.target_tickers == op1.target_tickers

    def test_event_sequence_dimension_tracking(self):
        """event_sequence correctly tracks universe size through Mode III events."""
        n_init = 3
        d = 5
        s0 = np.zeros(n_init * d)
        s0[P] = np.log(100)

        operators = [
            rate_change_operator(change_bps=25, n=3),   # Mode II, 3→3
            ipo_operator(ticker='NEW', ipo_price=50.0, n=3),  # Mode III, 3→4
            bankruptcy_operator(asset_idx=1, n=4, recovery_rate=0.05),  # Mode III, 4→3
        ]

        final_state, log = event_sequence(operators, s0, rng=np.random.default_rng(2026))
        # 3 → 3 → 4 → 3: net unchanged
        assert final_state.shape == (n_init * d,)
        assert len(log) == len(operators)


class TestMergerPremiumIndexing:
    """Regression tests for merger_operator b_w indexing (Bug #1)."""

    def test_merger_premium_follows_acquirer_index(self):
        """Deal premium must be applied at the acquirer's output row, not row 0."""
        n, d = 5, 5
        acquirer_idx, target_idx = 2, 4
        premium_pct = 30.0
        op = merger_operator(
            acquirer_idx=acquirer_idx, target_idx=target_idx,
            n=n, premium_pct=premium_pct,
        )
        # Acquirer (idx 2) should map to output row 2:
        # output rows: 0->0, 1->1, 2(acq)->2, 3->3, 4(tgt)->skipped
        acq_out_idx = 2
        expected_premium = np.log(1 + premium_pct / 100)

        # Premium should be at acquirer output row
        np.testing.assert_allclose(
            op.b_w[acq_out_idx * d + P], expected_premium, rtol=1e-6,
            err_msg="Premium should be at acquirer output row",
        )
        # Row 0 should have zero price shift (it's an uninvolved pass-through asset)
        np.testing.assert_allclose(
            op.b_w[0 * d + P], 0.0, atol=1e-10,
            err_msg="Row 0 should have no premium when acquirer_idx != 0",
        )

    def test_merger_premium_at_zero_unchanged(self):
        """When acquirer_idx=0, behavior is unchanged (backward compatibility)."""
        op = merger_operator(acquirer_idx=0, target_idx=1, n=3, premium_pct=25.0)
        expected = np.log(1.25)
        np.testing.assert_allclose(op.b_w[0 * 5 + P], expected, rtol=1e-6)

    def test_merger_volume_and_leverage_follow_acquirer(self):
        """Volume spike and leverage shift must also follow the acquirer row."""
        n, d = 4, 5
        op = merger_operator(acquirer_idx=2, target_idx=0, n=n)
        # Output mapping: 0(tgt)->skip, 1->0, 2(acq)->1, 3->2
        acq_out_idx = 1
        assert op.b_w[acq_out_idx * d + V] > 0, "Volume spike should be at acquirer row"
        assert op.b_w[acq_out_idx * d + L] > 0, "Leverage shift should be at acquirer row"
        # Other rows should have no volume/leverage shifts from merger
        for row in [0, 2]:
            assert op.b_w[row * d + V] == 0.0, f"Row {row} should have no volume shift"
            assert op.b_w[row * d + L] == 0.0, f"Row {row} should have no leverage shift"


class TestSystemicCrisisCorrelation:
    """Regression tests for systemic_crisis_operator Sigma_w (Bug #2)."""

    def test_systemic_crisis_has_cross_asset_correlation(self):
        """Crisis Sigma_w must have non-zero off-diagonal entries in the price block."""
        n = 4
        op = systemic_crisis_operator(severity=0.8, n=n)
        S = op.Sigma_w
        # The implied covariance Cov = S @ S.T should have non-zero off-diags
        Cov = S @ S.T
        price_indices = [i * 5 + P for i in range(n)]
        for i in price_indices:
            for j in price_indices:
                if i != j:
                    assert abs(Cov[i, j]) > 1e-6, \
                        f"Cov[{i},{j}] should be non-zero (crisis = correlated moves)"

    def test_systemic_crisis_covariance_is_psd(self):
        """Implied covariance must be PSD and preserve component variances."""
        severity = 1.0
        op = systemic_crisis_operator(severity=severity, n=5)
        Cov = op.Sigma_w @ op.Sigma_w.T
        eigs = np.linalg.eigvalsh(Cov)
        assert eigs.min() >= -1e-10, f"Min eigenvalue {eigs.min()} — covariance not PSD"
        np.testing.assert_allclose(np.diag(Cov), (severity * 0.08) ** 2)

    def test_systemic_crisis_correlation_increases_with_severity(self):
        """Higher severity should produce higher cross-asset correlation."""
        n = 3
        def avg_price_corr(severity):
            op = systemic_crisis_operator(severity=severity, n=n)
            Cov = op.Sigma_w @ op.Sigma_w.T
            pidx = [i * 5 + P for i in range(n)]
            stds = np.sqrt(np.array([Cov[i, i] for i in pidx]))
            corrs = []
            for a in range(n):
                for b in range(a + 1, n):
                    corrs.append(Cov[pidx[a], pidx[b]] / (stds[a] * stds[b] + 1e-12))
            return np.mean(corrs)

        rho_low = avg_price_corr(0.3)
        rho_high = avg_price_corr(0.9)
        assert rho_high > rho_low, \
            f"Higher severity should give higher correlation: {rho_high:.3f} vs {rho_low:.3f}"

    def test_systemic_crisis_single_asset(self):
        """n=1 should not crash (no off-diagonal terms needed)."""
        op = systemic_crisis_operator(severity=0.5, n=1)
        assert op.Sigma_w.shape == (5, 5)


class TestCompositionNoisePropagation:
    """
    Regression tests for the Sigma_w propagation law in compose().

    Sigma_w is documented as a Cholesky FACTOR, and every other test in this
    file reads its covariance as `S @ S.T`. compose() used `S @ S` — which
    agrees only when S is diagonal. Every composition test above uses
    diagonal-noise operators, and systemic_crisis_operator (the one
    constructor with off-diagonal noise) was never composed, so the gap
    stayed invisible.

    The failure mode was not merely a wrong covariance: for the crisis
    operator `S @ S` is not PSD, so np.linalg.cholesky raised LinAlgError and
    a systemic crisis could not be composed with any other event at all.
    """

    @staticmethod
    def _cov(op):
        return op.Sigma_w @ op.Sigma_w.T

    def test_composition_covariance_matches_propagation_law(self):
        """
        Cov(A1(A2 s + L2 ε2) + L1 ε1) = L1 L1ᵀ + A1 (L2 L2ᵀ) A1ᵀ.

        Uses systemic_crisis as the INNER operator so its off-diagonal block
        is the thing being propagated.
        """
        n = 3
        op_crisis = systemic_crisis_operator(severity=0.8, n=n)
        op_rate = rate_change_operator(change_bps=50, n=n)

        comp = compose(op_rate, op_crisis)  # crisis first, then rate change

        expected = (
            self._cov(op_rate)
            + op_rate.A_w @ self._cov(op_crisis) @ op_rate.A_w.T
        )
        np.testing.assert_allclose(self._cov(comp), expected, atol=1e-8)

    def test_composition_preserves_cross_asset_correlation(self):
        """
        The economic content: composing a macro event onto a systemic crisis
        must not destroy the crisis's correlated-price structure. `S @ S`
        silently reshapes those off-diagonals.
        """
        n = 4
        comp = compose(
            rate_change_operator(change_bps=25, n=n),
            systemic_crisis_operator(severity=0.9, n=n),
        )
        Cov = self._cov(comp)
        pidx = [i * 5 + P for i in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                assert Cov[pidx[a], pidx[b]] > 1e-6, (
                    f"price correlation between assets {a},{b} lost in composition"
                )

    def test_composed_sigma_is_a_valid_cholesky_factor(self):
        """Result must be lower-triangular so it composes again under the same law."""
        comp = compose(
            systemic_crisis_operator(severity=0.6, n=3),
            rate_change_operator(change_bps=25, n=3),
        )
        S = comp.Sigma_w
        np.testing.assert_allclose(S, np.tril(S), atol=1e-12)
        assert np.linalg.eigvalsh(self._cov(comp)).min() >= -1e-10

    def test_composition_is_associative_in_covariance(self):
        """
        Three-way composition must give the same covariance either way it is
        bracketed — the property that makes the monoid claim in the module
        docstring meaningful, and the one an S@S law breaks.
        """
        n = 3
        a = rate_change_operator(change_bps=25, n=n)
        b = systemic_crisis_operator(severity=0.7, n=n)
        c = earnings_shock_operator(surprise_pct=8.0, asset_idx=1, n=n)

        left = compose(compose(a, b), c)
        right = compose(a, compose(b, c))
        np.testing.assert_allclose(self._cov(left), self._cov(right), atol=1e-7)

    def test_diagonal_noise_case_is_unchanged(self):
        """
        Guards against over-correction: for diagonal Sigma the old and new laws
        agree, so previously-correct behaviour must not move.
        """
        op1 = stock_split_operator(ratio=2.0, n=1)
        op2 = dividend_operator(div_yield=0.02, asset_idx=0, n=1)
        comp = compose(op1, op2)
        expected = self._cov(op1) + op1.A_w @ self._cov(op2) @ op1.A_w.T
        np.testing.assert_allclose(self._cov(comp), expected, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
