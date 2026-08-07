"""
Tests for dual-noise decomposition (state/noise.py).
Validates BPV estimator, Lee-Mykland detection, and calibration.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state.noise import DualNoiseCalibrator, DualNoiseParams


class TestBiPowerVariation:
    def test_pure_brownian_bpv_equals_qv(self):
        """Without jumps, BPV ≈ QV (both estimate integrated variance)."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 1000
        returns = sigma * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        qv = np.sum(returns ** 2)

        # BPV should be close to QV for pure Brownian (ratio near 1)
        ratio = bpv / qv
        assert 0.85 < ratio < 1.15, f"BPV/QV ratio {ratio:.3f} outside [0.85, 1.15]"

    def test_bpv_lower_than_qv_with_jumps(self):
        """With jumps, QV > BPV (jumps inflate QV but not BPV)."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 500
        returns = sigma * rng.standard_normal(n)

        # Add large jumps
        jump_indices = rng.choice(n, size=10, replace=False)
        returns[jump_indices] += rng.choice([-0.05, 0.05], size=10)

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        qv = np.sum(returns ** 2)

        assert bpv < qv, f"BPV ({bpv:.6f}) should be < QV ({qv:.6f}) with jumps"

    def test_bpv_empty_returns(self):
        """Edge case: empty or single-element array."""
        cal = DualNoiseCalibrator()
        assert cal.estimate_bpv(np.array([])) == 0.0
        assert cal.estimate_bpv(np.array([0.01])) == 0.0

    def test_bpv_scales_with_variance(self):
        """BPV should scale linearly with σ²."""
        rng = np.random.default_rng(42)
        n = 2000
        r1 = 0.01 * rng.standard_normal(n)
        r2 = 0.02 * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        bpv1 = cal.estimate_bpv(r1)
        bpv2 = cal.estimate_bpv(r2)

        # BPV2 should be ~4x BPV1 (variance ratio = (0.02/0.01)^2 = 4)
        ratio = bpv2 / bpv1
        assert 3.0 < ratio < 5.0, f"BPV scaling ratio {ratio:.2f} outside [3, 5]"


class TestJumpDetection:
    def test_no_false_positives_in_pure_brownian(self):
        """Jump detector should rarely fire on pure Brownian paths."""
        rng = np.random.default_rng(42)
        sigma = 0.015
        n = 390  # ~1 day of 1-minute returns
        returns = sigma / np.sqrt(390) * rng.standard_normal(n)

        cal = DualNoiseCalibrator(alpha_lm=0.001)
        bpv = cal.estimate_bpv(returns)
        jumps = cal.detect_jumps(returns, bpv, dt=1/390)

        false_positive_rate = jumps.sum() / n
        assert false_positive_rate < 0.05, \
            f"False positive rate {false_positive_rate:.2%} too high (expected < 5%)"

    def test_detects_obvious_jumps(self):
        """Detector should find 10-sigma moves."""
        rng = np.random.default_rng(42)
        sigma = 0.01
        n = 500
        returns = sigma * rng.standard_normal(n)

        # Inject obvious jumps
        jump_indices = [100, 200, 350]
        for idx in jump_indices:
            returns[idx] = 0.15  # ~15σ move

        cal = DualNoiseCalibrator()
        bpv = cal.estimate_bpv(returns)
        jumps = cal.detect_jumps(returns, bpv, dt=1/78)

        detected = sum(jumps[idx] for idx in jump_indices)
        assert detected >= 2, f"Only {detected}/3 obvious jumps detected"


class TestCalibration:
    def test_calibration_returns_valid_params(self):
        """Calibration should return non-negative, finite parameters."""
        rng = np.random.default_rng(42)
        returns = 0.01 * rng.standard_normal(500)
        returns[50] = 0.08  # one jump

        cal = DualNoiseCalibrator()
        params = cal.calibrate(returns, dt=1/78)

        assert isinstance(params, DualNoiseParams)
        assert params.sigma_tau >= 0, "sigma_tau should be non-negative"
        assert np.isfinite(params.sigma_tau), "sigma_tau should be finite"
        assert params.lambda_eta >= 0, "lambda_eta should be non-negative"
        assert params.m2_eta > 0, "m2_eta should be positive"

    def test_tau_t_positive(self):
        """Composite temperature parameter τ_t should always be positive."""
        params = DualNoiseParams(sigma_tau=0.01, lambda_eta=2.0, m2_eta=0.001)
        assert params.tau_t > 0

    def test_higher_vol_gives_higher_sigma_tau(self):
        """Higher volatility data should give higher estimated sigma_tau."""
        rng = np.random.default_rng(42)
        n = 1000

        returns_low = 0.005 * rng.standard_normal(n)
        returns_high = 0.020 * rng.standard_normal(n)

        cal = DualNoiseCalibrator()
        params_low = cal.calibrate(returns_low)
        params_high = cal.calibrate(returns_high)

        assert params_high.sigma_tau > params_low.sigma_tau, \
            "Higher vol should give higher sigma_tau"


class TestCalibrationUnits:
    """
    Regression tests for the per-unit-time scaling in calibrate().

    The prior code computed `lambda_eta = N_jumps * dt`, which is the
    reciprocal of a rate: it under-reported intensity by a factor of n·dt²
    and — the giveaway — did not change when the sample got longer. Nothing
    caught it because the only assertion on lambda_eta was `>= 0`, which a
    dimensionally inverted quantity satisfies just fine.

    These tests pin the SEMANTICS (λ is a rate), not a magic number.
    """

    @staticmethod
    def _planted(n_days, bars_per_day, jumps_per_day, seed=7):
        """Quiet Brownian bars with `jumps_per_day` unmistakable jumps planted."""
        rng = np.random.default_rng(seed)
        n = n_days * bars_per_day
        r = rng.normal(0, 0.0005, n)
        idx = np.linspace(0, n - 1, n_days * jumps_per_day).astype(int)
        r[idx] += 0.05
        return r, len(set(idx))

    def test_lambda_eta_recovers_planted_intensity(self):
        """λ_η must come back in jumps per day, not jumps × dt."""
        bars = 78
        r, n_jumps = self._planted(n_days=10, bars_per_day=bars, jumps_per_day=3)

        params = DualNoiseCalibrator().calibrate(r, dt=1 / bars)

        # 10 days, 3 planted jumps/day → ~3/day. Detector may miss a few, so
        # allow a band — but the old code returned 0.038 here, 80x low.
        assert 2.0 < params.lambda_eta < 4.0, (
            f"λ_η = {params.lambda_eta:.4f}, expected ≈ {n_jumps / 10:.1f} jumps/day"
        )

    def test_lambda_eta_is_invariant_to_sample_length(self):
        """A rate is a property of the process, not of how long you watched it."""
        bars = 78
        short, _ = self._planted(n_days=2, bars_per_day=bars, jumps_per_day=3)
        long, _ = self._planted(n_days=20, bars_per_day=bars, jumps_per_day=3)

        cal = DualNoiseCalibrator()
        lam_short = cal.calibrate(short, dt=1 / bars).lambda_eta
        lam_long = cal.calibrate(long, dt=1 / bars).lambda_eta

        assert abs(lam_short - lam_long) < 1.0, (
            f"intensity drifted with sample length: {lam_short:.3f} vs {lam_long:.3f}"
        )

    def test_same_jump_count_over_longer_window_is_a_lower_rate(self):
        """
        Holds jump COUNT fixed and varies elapsed time — the one comparison a
        count-shaped estimator cannot fake. 6 jumps in 2 days is 3/day;
        6 jumps in 20 days is 0.3/day. `N_jumps * dt` returns the same number
        for both, because neither N nor dt changed.
        """
        bars = 78
        rng = np.random.default_rng(3)

        def six_jumps_over(n_days):
            n = n_days * bars
            r = rng.normal(0, 0.0005, n)
            r[np.linspace(0, n - 1, 6).astype(int)] += 0.05
            return r

        cal = DualNoiseCalibrator()
        lam_dense = cal.calibrate(six_jumps_over(2), dt=1 / bars).lambda_eta
        lam_sparse = cal.calibrate(six_jumps_over(20), dt=1 / bars).lambda_eta

        assert lam_dense > 5 * lam_sparse, (
            f"same 6 jumps over 2d vs 20d gave {lam_dense:.3f} vs {lam_sparse:.3f} "
            f"— λ_η is tracking count, not rate"
        )

    def test_sigma_tau_is_per_unit_time_not_per_bar(self):
        """
        σ_τ shares the clock with λ_η, so it scales as BPV/(n·dt). Sampling the
        SAME process at 5-min vs 1-min bars must give the same daily σ_τ; the
        old per-bar estimate `sqrt(BPV/n)` differed by √5 between them.
        """
        rng = np.random.default_rng(11)
        daily_sigma = 0.02

        cal = DualNoiseCalibrator()
        est = {}
        for bars in (78, 390):  # 5-min and 1-min bars over one day
            r = rng.normal(0, daily_sigma / np.sqrt(bars), bars * 5)
            est[bars] = cal.calibrate(r, dt=1 / bars).sigma_tau

        np.testing.assert_allclose(est[78], est[390], rtol=0.25)
        np.testing.assert_allclose(est[78], daily_sigma, rtol=0.25)

    def test_cramer_rao_bound_is_invariant_to_sampling_frequency(self):
        """
        Theorem III.1 is a statement about a PROCESS, so observing that process
        on 5-min vs 1-min bars must yield the same 1-day bound. This is the
        end-to-end check that σ_τ² and λ_η·m₂^η share a clock: under the old
        scaling σ_τ² moved by 5x and λ_η by 25x between these two samplings,
        in opposite directions.
        """
        cal = DualNoiseCalibrator()
        bound = {}
        for bars in (78, 390):
            rng = np.random.default_rng(5)
            n = bars * 10
            r = rng.normal(0, 0.02 / np.sqrt(bars), n)
            r[np.linspace(0, n - 1, 30).astype(int)] += 0.05  # 3 jumps/day
            bound[bars] = cal.calibrate(r, dt=1 / bars).cramer_rao_bound(h=1.0)

        np.testing.assert_allclose(bound[78], bound[390], rtol=0.35)

    def test_empty_and_degenerate_input(self):
        """Guard the early return added alongside the rescaling."""
        cal = DualNoiseCalibrator()
        for returns, dt in [(np.array([]), 1 / 78), (np.array([0.01, 0.02]), 0.0)]:
            p = cal.calibrate(returns, dt=dt)
            assert np.isfinite(p.sigma_tau) and np.isfinite(p.lambda_eta)
            assert p.m2_eta > 0


class TestCramerRaoBound:
    """Regression tests for DualNoiseParams.cramer_rao_bound (Bug #3)."""

    def test_cramer_rao_bound_callable_with_h(self):
        """cramer_rao_bound must accept explicit horizon parameter h."""
        params = DualNoiseParams(sigma_tau=0.01, lambda_eta=2.0, m2_eta=0.001)
        # Should not raise TypeError
        result = params.cramer_rao_bound(h=5.0)
        assert isinstance(result, float)
        assert result > 0

    def test_cramer_rao_bound_scales_with_horizon(self):
        """Bound at horizon h should be h times the bound at horizon 1."""
        params = DualNoiseParams(sigma_tau=0.01, lambda_eta=2.0, m2_eta=0.001)
        bound_1 = params.cramer_rao_bound(h=1.0)
        bound_5 = params.cramer_rao_bound(h=5.0)
        np.testing.assert_allclose(bound_5, 5.0 * bound_1, rtol=1e-10)

    def test_cramer_rao_bound_default_h(self):
        """Default h=1.0 should produce the same result as explicit h=1.0."""
        params = DualNoiseParams(sigma_tau=0.02, lambda_eta=1.0, m2_eta=0.005)
        np.testing.assert_allclose(
            params.cramer_rao_bound(),
            params.cramer_rao_bound(h=1.0),
        )

    def test_cramer_rao_bound_matches_formula(self):
        """Verify against Theorem III.1: (σ_τ² + λ_η · m₂^η) · h."""
        params = DualNoiseParams(sigma_tau=0.03, lambda_eta=5.0, m2_eta=0.01)
        h = 3.0
        expected = (0.03**2 + 5.0 * 0.01) * h
        np.testing.assert_allclose(params.cramer_rao_bound(h=h), expected, rtol=1e-10)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
