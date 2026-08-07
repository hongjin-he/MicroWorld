"""
Financial event operators — complete matrix formalization (§IV).

State vector per asset: s = [p, v, ℓ, κ, ι] ∈ ℝ^d, d = 5
  p = log price
  v = log trading volume
  ℓ = log leverage (debt-to-equity)
  κ = log shares outstanding (market cap proxy)
  ι = information disclosure level (0=opaque, 1=transparent)

Affine event operator:
  T_w(s) = A_w · s + b_w + Σ_w ε_w,   ε_w ~ N(0, I)

Three algebraic modes:
  Mode I   — Local endomorphism:  T_w : ℝ^(n·d) → ℝ^(n·d),  n preserved
              Single-asset or whole-market events that keep universe size fixed
              Examples: stock split, dividend, buyback, earnings shock, analyst rating

  Mode II  — Global tensor product: T_w : ℝ^(n·d) → ℝ^(n·d),  n preserved
              Macro events that simultaneously affect ALL assets via a shared factor
              Examples: rate hike/cut, QE/QT, circuit breaker, systemic crisis

  Mode III — Pairwise morphism: T_w : ℝ^(n·d) → ℝ^(m·d),  m ≠ n
              Corporate structure changes that alter the investment universe
              Examples: IPO (n→n+1), merger (n→n-1), delisting (n→n-1), spin-off (n→n+1)

Algebraic structure:
  Mode I + II operators form a MONOID under composition (fixed domain ℝ^(n·d)).
  All three modes together form a GROUPOID — composition is defined only when
  the target dimension of the right operator matches the source dimension of the left.

  compose(T1, T2) defined iff dim(target(T2)) == dim(source(T1))
  A_comp = A1 @ A2
  b_comp = A1 @ b2 + b1
  Σ_comp = √(Σ1² + (A1 @ Σ2)²)  [propagation of uncertainty]

Reference: Alpha Flow §IV; Barndorff-Nielsen & Shephard (2004) for noise structure.
"""
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple


# ── State vector indices ───────────────────────────────────────────────────────
P = 0   # log price
V = 1   # log volume
L = 2   # log leverage
K = 3   # log shares outstanding
I = 4   # information disclosure


class EventMode(Enum):
    LOCAL   = "I"    # single asset or whole-market, dimension-preserving
    GLOBAL  = "II"   # macro event, all assets simultaneously
    PAIRWISE = "III" # changes number of assets


@dataclass
class EventOperator:
    """
    Affine operator T_w(s) = A_w · s + b_w + Σ_w ε_w

    Attributes
    ----------
    name         : human-readable event label
    mode         : algebraic mode (I / II / III)
    A_w          : (m·d, n·d) transformation matrix
    b_w          : (m·d,) additive shift
    Sigma_w      : (m·d, m·d) noise covariance (lower-triangular Cholesky)
    source_dim   : n (number of source assets)
    target_dim   : m (number of target assets)
    source_tickers : ordered list of input asset identifiers
    target_tickers : ordered list of output asset identifiers
    """
    name: str
    mode: EventMode
    A_w: np.ndarray
    b_w: np.ndarray
    Sigma_w: np.ndarray
    source_dim: int
    target_dim: int
    source_tickers: List[str] = field(default_factory=list)
    target_tickers: List[str] = field(default_factory=list)

    @property
    def d(self) -> int:
        """State dimension per asset."""
        return self.A_w.shape[0] // self.target_dim if self.target_dim > 0 else 5

    def apply(self, s: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Apply T_w to state vector s.

        Parameters
        ----------
        s   : (n·d,) source state
        rng : random number generator (default: new default_rng)
        """
        rng = rng or np.random.default_rng()
        eps = rng.standard_normal(self.b_w.shape)
        return self.A_w @ s + self.b_w + self.Sigma_w @ eps

    def matrix_summary(self) -> str:
        """Print A_w, b_w shapes and key non-zero entries."""
        lines = [
            f"EventOperator: {self.name}",
            f"  Mode: {self.mode.value}  |  {self.source_dim}→{self.target_dim} assets",
            f"  A_w shape: {self.A_w.shape}",
            f"  b_w shape: {self.b_w.shape}",
            f"  Sigma_w shape: {self.Sigma_w.shape}",
            f"  |b_w|_inf = {np.abs(self.b_w).max():.4f}",
            f"  |A_w - I|_F = {np.linalg.norm(self.A_w - np.eye(*self.A_w.shape) if self.A_w.shape[0]==self.A_w.shape[1] else self.A_w):.4f}",
        ]
        return "\n".join(lines)


def _identity_local(n: int, d: int = 5) -> np.ndarray:
    """Block-diagonal identity for n assets of dimension d."""
    return np.eye(n * d)


# ══════════════════════════════════════════════════════════════════════════════
# MODE I — Local endomorphisms (dimension-preserving, single-asset events)
# ══════════════════════════════════════════════════════════════════════════════

def stock_split_operator(ratio: float, asset_idx: int = 0, n: int = 1, d: int = 5) -> EventOperator:
    """
    k-for-1 stock split on asset `asset_idx`.

    Matrix action on asset i:
      p_i → p_i - log(k)    [price halves for k=2]
      κ_i → κ_i + log(k)    [shares outstanding doubles]
      v, ℓ, ι unchanged

    A_w = I_{nd},  b_w[i·d + P] = -log(k),  b_w[i·d + K] = +log(k)
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = -np.log(ratio)
    b[asset_idx * d + K] = +np.log(ratio)
    return EventOperator(
        name=f"stock_split_{ratio}x_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.zeros((n * d, n * d)),
        source_dim=n, target_dim=n,
    )


def reverse_split_operator(ratio: float, asset_idx: int = 0, n: int = 1, d: int = 5) -> EventOperator:
    """
    1-for-k reverse split (consolidation).

    p_i → p_i + log(k),  κ_i → κ_i - log(k)
    Often precedes delisting — company tries to stay above exchange minimum price.
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = +np.log(ratio)
    b[asset_idx * d + K] = -np.log(ratio)
    return EventOperator(
        name=f"reverse_split_{ratio}x_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.zeros((n * d, n * d)),
        source_dim=n, target_dim=n,
    )


def dividend_operator(div_yield: float, asset_idx: int = 0, n: int = 1, d: int = 5) -> EventOperator:
    """
    Ex-dividend price adjustment.

    On ex-date: p_i → p_i - log(1 + div_yield)
    Exact for log prices: log(S - D) ≈ log(S) - D/S = log(S) - div_yield
    No change to shares outstanding κ (cash dividend, not stock dividend).

    Noise: small because ex-date is known in advance.
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = -np.log(1 + div_yield)
    return EventOperator(
        name=f"dividend_{div_yield:.2%}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([1e-4 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def secondary_offering_operator(dilution_frac: float, asset_idx: int = 0,
                                 n: int = 1, d: int = 5) -> EventOperator:
    """
    Secondary equity offering (SEO): new shares issued at discount.

    κ_i → κ_i + log(1 + dilution_frac)    [shares up]
    p_i → p_i + log(1 - dilution_frac/2)  [price down ~half dilution, empirical]
    ℓ_i → ℓ_i - log(1 + dilution_frac)   [leverage decreases as equity base grows]
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = np.log(1 - dilution_frac / 2)
    b[asset_idx * d + L] = -np.log(1 + dilution_frac)
    b[asset_idx * d + K] = np.log(1 + dilution_frac)
    return EventOperator(
        name=f"secondary_offering_{dilution_frac:.0%}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.02 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def share_buyback_operator(buyback_frac: float, asset_idx: int = 0,
                            n: int = 1, d: int = 5) -> EventOperator:
    """
    Share repurchase: company buys back fraction `buyback_frac` of outstanding shares.

    κ_i → κ_i + log(1 - buyback_frac)   [shares outstanding down]
    p_i → p_i + log(1 + buyback_frac)   [EPS accretion → price up]
    ℓ_i → ℓ_i + log(1 + buyback_frac * 0.5)  [slight leverage increase, cash used]

    Dual: opposite of secondary_offering.
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = np.log(1 + buyback_frac)
    b[asset_idx * d + L] = np.log(1 + buyback_frac * 0.5)
    b[asset_idx * d + K] = np.log(1 - buyback_frac)
    return EventOperator(
        name=f"buyback_{buyback_frac:.0%}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.01 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def earnings_shock_operator(surprise_pct: float, asset_idx: int = 0,
                              n: int = 1, d: int = 5) -> EventOperator:
    """
    Earnings announcement: price jumps by EPS surprise.

    Empirically (Ball-Brown 1968, PEAD literature):
      p_i → p_i + β · surprise_pct   [β ≈ 0.03 for 1% EPS surprise → 3% price move]
      v_i → v_i + log(3)             [volume spikes 3x on announcement]
      ι_i → ι_i + Δι                [information set expands post-announcement]

    Noise large: earnings reactions are notoriously unpredictable.
    """
    beta_price = 0.03  # 1% EPS surprise → 3% price move (stylized fact)
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = beta_price * surprise_pct / 100
    b[asset_idx * d + V] = np.log(3.0)   # 3x volume
    b[asset_idx * d + I] = 0.15          # information disclosure spike
    return EventOperator(
        name=f"earnings_shock_{surprise_pct:+.1f}pct_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.03 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def analyst_rating_operator(rating_change: int, asset_idx: int = 0,
                              n: int = 1, d: int = 5) -> EventOperator:
    """
    Analyst rating change: upgrade (+1) or downgrade (-1).

    Empirical magnitudes (Womack 1996):
      Upgrade (+1):   p ≈ +3%, v ≈ +80%
      Downgrade (-1): p ≈ -4%, v ≈ +150%
    Asymmetric: downgrades have larger price impact than upgrades.
    """
    if rating_change > 0:
        dp, dv = 0.03, np.log(1.8)
    else:
        dp, dv = -0.04, np.log(2.5)

    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = dp * abs(rating_change)
    b[asset_idx * d + V] = dv
    return EventOperator(
        name=f"analyst_{'upgrade' if rating_change > 0 else 'downgrade'}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.015 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def index_change_operator(is_inclusion: bool, asset_idx: int = 0,
                           n: int = 1, d: int = 5) -> EventOperator:
    """
    Index inclusion or exclusion (e.g., SP500 rebalance).

    Inclusion: forced buying from passive → p up, v up, ι up
    Exclusion: forced selling from passive → p down, v up

    Empirical (Shleifer 1986, Harris & Gurel 1986):
      Inclusion: ~3.5% abnormal return around announcement
      Exclusion: ~-2.5% abnormal return
    """
    sign = +1 if is_inclusion else -1
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + P] = sign * 0.035
    b[asset_idx * d + V] = np.log(2.0)
    b[asset_idx * d + I] = 0.05 if is_inclusion else 0.0
    return EventOperator(
        name=f"index_{'inclusion' if is_inclusion else 'exclusion'}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=np.diag([0.01 if j == asset_idx * d + P else 0 for j in range(n * d)]),
        source_dim=n, target_dim=n,
    )


def trading_halt_operator(halt_duration_hours: float, asset_idx: int = 0,
                           n: int = 1, d: int = 5) -> EventOperator:
    """
    Regulatory trading halt (news pending, circuit breaker, SEC halt).

    During halt: volume → 0 (hard constraint via large negative shift in log volume)
    Post-halt: information set expands significantly (ι up)
    Price uncertainty increases (noise spike on reopening).
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    b[asset_idx * d + V] = -10.0   # volume effectively zero
    b[asset_idx * d + I] = 0.30    # information event (why was it halted?)
    # Reopen with elevated uncertainty
    Sigma = np.zeros((n * d, n * d))
    Sigma[asset_idx * d + P, asset_idx * d + P] = 0.05 * np.sqrt(halt_duration_hours)
    return EventOperator(
        name=f"trading_halt_{halt_duration_hours:.1f}h_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b,
        Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def short_squeeze_operator(squeeze_intensity: float, asset_idx: int = 0,
                            n: int = 1, d: int = 5) -> EventOperator:
    """
    Short squeeze: forced covering by shorts amplifies upward price move.

    Mechanism: short interest high → price rises → shorts forced to cover → more buying
    A_w: price has positive feedback coefficient > 1 during squeeze
    This is a TEMPORARY non-linearity; we approximate as large positive b + A amplification.

    squeeze_intensity ∈ [0, 1]: 1 = GME-level squeeze
    """
    A = _identity_local(n, d)
    # Amplify price momentum during squeeze
    A[asset_idx * d + P, asset_idx * d + P] = 1.0 + squeeze_intensity * 0.5
    b = np.zeros(n * d)
    b[asset_idx * d + P] = squeeze_intensity * 0.30   # price spike
    b[asset_idx * d + V] = np.log(1 + squeeze_intensity * 10)  # massive volume
    Sigma = np.zeros((n * d, n * d))
    Sigma[asset_idx * d + P, asset_idx * d + P] = squeeze_intensity * 0.08
    return EventOperator(
        name=f"short_squeeze_intensity{squeeze_intensity:.1f}_asset{asset_idx}",
        mode=EventMode.LOCAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODE II — Global tensor product (all assets simultaneously)
# ══════════════════════════════════════════════════════════════════════════════

def rate_change_operator(change_bps: float, n: int = 1, d: int = 5,
                          duration_profile: Optional[np.ndarray] = None) -> EventOperator:
    """
    Central bank rate change (+bps = hike, -bps = cut).

    Duration-weighted price impact per asset:
      p_i → p_i - duration_i · (change_bps / 10000)

    Default duration_profile: uniform duration = 3.5 years.
    Leverage increases on hike (financing cost), decreases on cut.
    Volume spikes on announcement regardless of direction.
    """
    if duration_profile is None:
        duration_profile = np.full(n, 3.5)  # years

    rate_delta = change_bps / 10000
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + P] = -duration_profile[i] * rate_delta
        b[i * d + L] = +rate_delta * 0.5     # financing cost → leverage up on hike
        b[i * d + V] = np.log(1.5)            # volume spike on announcement

    direction = 'hike' if change_bps > 0 else 'cut'
    Sigma = np.eye(n * d) * (abs(change_bps) / 10000 * 0.3)
    # Cross-asset correlation in Sigma: macro events create correlated moves
    for i in range(n):
        for j in range(n):
            if i != j:
                Sigma[i * d + P, j * d + P] = abs(change_bps) / 10000 * 0.15
    return EventOperator(
        name=f"rate_{direction}_{abs(change_bps):.0f}bps",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def qe_operator(size_bn: float, n: int = 1, d: int = 5) -> EventOperator:
    """
    Quantitative easing: central bank buys assets → risk-on environment.

    Effect:
      - Long-duration assets benefit most (p up, duration-weighted)
      - Credit spreads compress → ℓ down (companies can refinance)
      - Risk appetite increases → all assets up, correlation increases
      - ι up: policy transparency (forward guidance announced)

    size_bn: QE size in $B (proxy for signal strength)
    """
    signal = np.log(1 + size_bn / 500) * 0.05  # diminishing returns
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + P] = signal
        b[i * d + L] = -signal * 0.3  # refinancing → deleverage
        b[i * d + I] = 0.1

    Sigma = np.eye(n * d) * (signal * 0.1)
    return EventOperator(
        name=f"qe_{size_bn:.0f}bn",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def qt_operator(size_bn: float, n: int = 1, d: int = 5) -> EventOperator:
    """
    Quantitative tightening: reverse of QE. Liquidity withdrawal.

    Opposite signs from qe_operator but asymmetric magnitudes
    (tightening impact > easing impact due to liquidity asymmetry).
    """
    signal = np.log(1 + size_bn / 500) * 0.07  # asymmetrically larger
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + P] = -signal
        b[i * d + L] = +signal * 0.4

    Sigma = np.eye(n * d) * (signal * 0.15)
    return EventOperator(
        name=f"qt_{size_bn:.0f}bn",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def systemic_crisis_operator(severity: float, n: int = 1, d: int = 5) -> EventOperator:
    """
    Systemic financial crisis (Lehman-level = severity 1.0).

    All assets suffer correlated crash. Key features:
      - Price: all down (correlation → 1 in crisis)
      - Volume: massive spike (panic selling)
      - Leverage: spikes (collateral calls, forced deleveraging)
      - Information: drops (uncertainty, opacity)
      - Cross-asset Sigma: near-singular (all move together)

    severity ∈ (0, 1]: 0.3 = flash crash, 0.7 = GFC 2008, 1.0 = hypothetical worst case
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + P] = -severity * 0.20       # -20% at peak (SPX ~ -50% intraday implied)
        b[i * d + V] = np.log(1 + severity * 5)  # 5x volume at peak
        b[i * d + L] = +severity * 0.30        # forced leverage
        b[i * d + I] = -severity * 0.20        # opacity spike

    # Near-singular covariance: correlations → 1 in crisis
    # Build the target covariance matrix for price components, then Cholesky-factor it.
    rho = 0.5 + 0.4 * severity
    sigma_p = severity * 0.08  # per-asset price std dev
    Cov = np.eye(n * d) * sigma_p ** 2
    for i in range(n):
        for j in range(n):
            if i != j:
                Cov[i * d + P, j * d + P] = sigma_p ** 2 * rho
    Sigma = np.linalg.cholesky(Cov)

    return EventOperator(
        name=f"systemic_crisis_severity{severity:.1f}",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def circuit_breaker_operator(n: int = 1, d: int = 5) -> EventOperator:
    """
    Market-wide circuit breaker (NYSE Rule 80B).

    Triggered at -7%, -13%, -20% SPX moves.
    During halt: volume → 0 for ALL assets simultaneously.
    Post-halt reopening: elevated uncertainty for all.
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + V] = -10.0  # volume zeroed
        b[i * d + I] = +0.10  # information event

    Sigma = np.eye(n * d) * 0.03  # reopen uncertainty
    return EventOperator(
        name="market_circuit_breaker",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def volatility_regime_shift_operator(new_regime_vol: float, old_regime_vol: float,
                                      n: int = 1, d: int = 5) -> EventOperator:
    """
    VIX regime shift: transition between low-vol and high-vol regimes.

    This is captured in the A_w matrix (multiplicative rescaling of vol):
      The noise covariance Sigma_w expands/contracts.
    No direct b_w effect on price — the regime shift affects future dynamics, not levels.
    """
    vol_ratio = new_regime_vol / max(old_regime_vol, 1e-6)
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    # Vol regime affects leverage (higher vol → margin calls → forced deleveraging)
    for i in range(n):
        if vol_ratio > 1:
            b[i * d + L] = np.log(vol_ratio) * 0.3

    Sigma = np.eye(n * d) * (new_regime_vol / 100)
    return EventOperator(
        name=f"vol_regime_shift_{old_regime_vol:.0f}to{new_regime_vol:.0f}",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


def inflation_shock_operator(cpi_surprise_pct: float, n: int = 1, d: int = 5) -> EventOperator:
    """
    Inflation surprise (CPI release beat/miss).

    High inflation surprise:
      - Nominal prices up (TIPS holders relieved, equities ambiguous)
      - Real rates rise → duration-sensitive assets down
      - Uncertainty spikes

    Empirically: equity markets react negatively to CPI beats (real rate effect dominates).
    """
    A = _identity_local(n, d)
    b = np.zeros(n * d)
    for i in range(n):
        b[i * d + P] = -cpi_surprise_pct / 100 * 0.5  # equity down on inflation beat
        b[i * d + V] = np.log(1.5)

    Sigma = np.eye(n * d) * abs(cpi_surprise_pct) / 100 * 0.2
    direction = 'beat' if cpi_surprise_pct > 0 else 'miss'
    return EventOperator(
        name=f"cpi_{direction}_{abs(cpi_surprise_pct):.1f}pct",
        mode=EventMode.GLOBAL,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=n,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODE III — Pairwise morphisms (universe size changes: n → m ≠ n)
# ══════════════════════════════════════════════════════════════════════════════

def merger_operator(acquirer_idx: int, target_idx: int, n: int = 2, d: int = 5,
                    premium_pct: float = 30.0,
                    acquirer_weight: float = 0.6) -> EventOperator:
    """
    M&A merger: n assets → (n-1) assets. Target absorbed into acquirer.

    Matrix structure: A_w ∈ ℝ^{(n-1)d × nd}
    For each component:
      p_combined = acquirer_weight·p_acq + (1-acquirer_weight)·p_tgt + log(1 + premium/100)
      v_combined = log(exp(v_acq) + exp(v_tgt))  ≈ max + log(2)  [pool volumes]
      ℓ_combined = acquirer_weight·ℓ_acq + (1-a)·ℓ_tgt + Δℓ_synergy
      κ_combined = log(exp(κ_acq) + exp(κ_tgt))  [market caps add]
      ι_combined = max(ι_acq, ι_tgt)  [better of two disclosure levels]

    All other assets pass through unchanged.
    """
    m = n - 1  # output universe size
    A = np.zeros((m * d, n * d))

    out_idx = 0
    for i in range(n):
        if i == acquirer_idx:
            # Acquirer row: weighted combination of acquirer + target states
            tw = 1 - acquirer_weight
            A[out_idx * d + P, acquirer_idx * d + P] = acquirer_weight
            A[out_idx * d + P, target_idx * d + P]   = tw
            A[out_idx * d + V, acquirer_idx * d + V] = 1.0
            A[out_idx * d + L, acquirer_idx * d + L] = acquirer_weight
            A[out_idx * d + L, target_idx * d + L]   = tw
            A[out_idx * d + K, acquirer_idx * d + K] = 1.0  # log of sum handled in b
            A[out_idx * d + I, acquirer_idx * d + I] = acquirer_weight
            A[out_idx * d + I, target_idx * d + I]   = tw
            out_idx += 1
        elif i == target_idx:
            pass  # target disappears from universe
        else:
            # Pass-through for other assets
            A[out_idx * d:(out_idx + 1) * d, i * d:(i + 1) * d] = np.eye(d)
            out_idx += 1

    # Compute acquirer's output row index
    acq_out_idx = 0
    for i in range(acquirer_idx):
        if i != target_idx:
            acq_out_idx += 1

    b = np.zeros(m * d)
    b[acq_out_idx * d + P] = np.log(1 + premium_pct / 100)  # deal premium on acquirer row
    b[acq_out_idx * d + V] = np.log(1.5)  # volume addition (approximate log-sum-exp)
    b[acq_out_idx * d + L] = 0.05  # slight leverage increase from deal financing

    Sigma = np.eye(m * d) * 0.05
    Sigma[acq_out_idx * d + P, acq_out_idx * d + P] = 0.08  # acquirer price most uncertain

    return EventOperator(
        name=f"merger_acq{acquirer_idx}_tgt{target_idx}_prem{premium_pct:.0f}pct",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=m,
        source_tickers=[str(i) for i in range(n)],
        target_tickers=[str(i) for i in range(m)],
    )


def spinoff_operator(parent_idx: int, n: int = 1, d: int = 5,
                     child_fraction: float = 0.30) -> EventOperator:
    """
    Corporate spin-off: n assets → (n+1) assets.

    Parent carves out a subsidiary, which begins trading as a separate entity.

    A_w ∈ ℝ^{(n+1)d × nd}:
      Parent retains (1 - child_fraction) of value
      Child starts at child_fraction of parent's state

    Empirical (Cusatis et al. 1993):
      Both parent and child outperform market post-spinoff (focus effect)
    """
    m = n + 1
    A = np.zeros((m * d, n * d))

    # Pass through all existing assets
    for i in range(n):
        A[i * d:(i + 1) * d, i * d:(i + 1) * d] = np.eye(d)

    # Parent: price haircut at spinoff (value transferred to child)
    pf = child_fraction
    A[parent_idx * d + P, parent_idx * d + P] = 1.0

    # Child (new asset at position n): inherits from parent
    A[n * d + P, parent_idx * d + P] = 1.0
    A[n * d + V, parent_idx * d + V] = 1.0
    A[n * d + L, parent_idx * d + L] = 0.8  # child typically less leveraged
    A[n * d + K, parent_idx * d + K] = 1.0
    A[n * d + I, parent_idx * d + I] = 0.7  # child less transparent initially

    b = np.zeros(m * d)
    b[parent_idx * d + P] = np.log(1 - pf)  # parent: loses child value
    b[n * d + P] = np.log(pf)               # child: starts at fraction of parent price
    b[n * d + P] += -0.20                   # discount: new, unknown entity
    b[n * d + V] = -np.log(3)              # lower initial liquidity

    Sigma = np.eye(m * d) * 0.05
    Sigma[n * d + P, n * d + P] = 0.12  # high uncertainty for new entity

    return EventOperator(
        name=f"spinoff_parent{parent_idx}_child_frac{pf:.0%}",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=m,
        source_tickers=[str(i) for i in range(n)],
        target_tickers=[str(i) for i in range(m)],
    )


def ipo_operator(ticker: str, ipo_price: float = 90.0,
                 n: int = 0, d: int = 5) -> EventOperator:
    """
    IPO: n assets → (n+1) assets. New company enters public market.

    A_w ∈ ℝ^{(n+1)d × nd}: identity for existing, zero row for new (initialised from b_w).

    IPO pricing heuristics:
      - Initial log_price = log(ipo_price)
      - Low volume initially (limited float)
      - High uncertainty (Sigma large for new entrant)
      - Low information ι (limited track record)

    Note: ipo_price should be the IPO offering price, not the opening trade price.
    """
    m = n + 1
    A = np.zeros((m * d, n * d)) if n > 0 else np.zeros((d, max(n * d, 1)))
    if n > 0:
        A[:n * d, :n * d] = np.eye(n * d)  # existing assets pass through

    b = np.zeros(m * d)
    b[n * d + P] = np.log(ipo_price)
    b[n * d + V] = 2.5    # moderate initial volume (limited float)
    b[n * d + L] = -0.5   # typical IPO: low leverage (just raised capital)
    b[n * d + K] = np.log(ipo_price * 1e6)  # market cap proxy
    b[n * d + I] = 0.3    # S-1 filing increases transparency

    Sigma = np.zeros((m * d, m * d))
    if n > 0:
        Sigma[:n * d, :n * d] = np.eye(n * d) * 1e-6  # existing assets: no change
    Sigma[n * d + P, n * d + P] = 0.15  # high price uncertainty on first day
    Sigma[n * d + V, n * d + V] = 0.30  # volume very uncertain

    return EventOperator(
        name=f"ipo_{ticker}_{ipo_price:.0f}",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=m,
        source_tickers=[str(i) for i in range(n)],
        target_tickers=[str(i) for i in range(n)] + [ticker],
    )


def delisting_operator(asset_idx: int, n: int = 2, d: int = 5) -> EventOperator:
    """
    Voluntary or involuntary delisting: n assets → (n-1) assets.

    Asset is removed from the universe. Price goes to zero (or last trade price),
    volume drops to zero, no further information updates.

    Differs from bankruptcy_operator in that voluntary delisting (going private,
    acquisition) may have positive price impact for remaining holders who
    receive buyout premium.
    """
    m = n - 1
    A = np.zeros((m * d, n * d))

    out_idx = 0
    for i in range(n):
        if i == asset_idx:
            continue  # asset removed from universe
        A[out_idx * d:(out_idx + 1) * d, i * d:(i + 1) * d] = np.eye(d)
        out_idx += 1

    b = np.zeros(m * d)
    Sigma = np.eye(m * d) * 1e-6  # no uncertainty for remaining assets
    return EventOperator(
        name=f"delisting_asset{asset_idx}",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=m,
        source_tickers=[str(i) for i in range(n)],
        target_tickers=[str(i) for i in range(n) if i != asset_idx],
    )


def bankruptcy_operator(asset_idx: int, n: int = 2, d: int = 5,
                         recovery_rate: float = 0.10) -> EventOperator:
    """
    Corporate bankruptcy (Chapter 11/7): n assets → (n-1) assets.

    Price goes to recovery_rate fraction of pre-filing price (senior debt recovery).
    Equity typically worthless (recovery_rate → 0 for equity holders).
    Volume spikes initially then collapses.

    Differs from delisting in the pre-removal price dynamics and recovery economics.
    """
    m = n - 1
    # Before removal: price drops to recovery rate
    A_with = _identity_local(n, d)
    A_with[asset_idx * d + P, asset_idx * d + P] = 1.0
    b_with = np.zeros(n * d)
    b_with[asset_idx * d + P] = np.log(recovery_rate)  # price to recovery rate
    b_with[asset_idx * d + V] = np.log(3.0)            # volume spike on filing

    # Then remove (same as delisting)
    A_rm = np.zeros((m * d, n * d))
    out_idx = 0
    for i in range(n):
        if i == asset_idx:
            continue
        A_rm[out_idx * d:(out_idx + 1) * d, i * d:(i + 1) * d] = np.eye(d)
        out_idx += 1

    A = A_rm @ A_with
    b = A_rm @ b_with
    Sigma = np.eye(m * d) * 0.01

    return EventOperator(
        name=f"bankruptcy_asset{asset_idx}_recovery{recovery_rate:.0%}",
        mode=EventMode.PAIRWISE,
        A_w=A, b_w=b, Sigma_w=Sigma,
        source_dim=n, target_dim=m,
        source_tickers=[str(i) for i in range(n)],
        target_tickers=[str(i) for i in range(n) if i != asset_idx],
    )


# ══════════════════════════════════════════════════════════════════════════════
# GROUPOID COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def compose(op1: EventOperator, op2: EventOperator) -> EventOperator:
    """
    Groupoid composition: T1 ∘ T2  (apply T2 first, then T1).

    Defined only when: dim(target(T2)) == dim(source(T1))
    i.e., op2.target_dim == op1.source_dim

    Composition law for affine operators:
      A_comp = A1 @ A2
      b_comp = A1 @ b2 + b1
      Σ_comp = √(Σ1² + (A1 @ Σ2)²)   [uncertainty propagation]

    Raises ValueError on domain mismatch.
    """
    if op2.target_dim != op1.source_dim:
        raise ValueError(
            f"Groupoid composition undefined: "
            f"op2 '{op2.name}' maps to {op2.target_dim} assets, "
            f"but op1 '{op1.name}' expects {op1.source_dim} assets as input. "
            f"Composition T1 ∘ T2 requires dim(target(T2)) = dim(source(T1))."
        )
    if op1.A_w.shape[1] != op2.A_w.shape[0]:
        raise ValueError(
            f"Matrix dimension mismatch for composition: "
            f"A1 {op1.A_w.shape}, A2 {op2.A_w.shape}"
        )

    A_comp = op1.A_w @ op2.A_w
    b_comp = op1.A_w @ op2.b_w + op1.b_w
    # Uncertainty propagation. Sigma_w is a Cholesky FACTOR, so the covariance
    # it represents is Σ Σᵀ — not Σ Σ. The two coincide only for diagonal Σ,
    # which is why this went unnoticed: systemic_crisis_operator is the one
    # constructor with off-diagonal noise, and nothing composed it. For that
    # operator `Σ Σ` is not even PSD, so the cholesky() below raised
    # LinAlgError — composing a systemic crisis with anything was impossible.
    #   Cov(A1(A2 s + L2 ε2) + L1 ε1) = L1 L1ᵀ + A1 (L2 L2ᵀ) A1ᵀ
    Cov_comp = (
        op1.Sigma_w @ op1.Sigma_w.T
        + op1.A_w @ (op2.Sigma_w @ op2.Sigma_w.T) @ op1.A_w.T
    )
    # Back to a Cholesky factor, so the result composes again under this same law.
    Sigma_comp = np.linalg.cholesky(Cov_comp + 1e-9 * np.eye(Cov_comp.shape[0]))

    return EventOperator(
        name=f"({op1.name}) ∘ ({op2.name})",
        mode=op2.mode,
        A_w=A_comp, b_w=b_comp, Sigma_w=Sigma_comp,
        source_dim=op2.source_dim, target_dim=op1.target_dim,
        source_tickers=op2.source_tickers,
        target_tickers=op1.target_tickers,
    )


def event_sequence(operators: list, initial_state: np.ndarray,
                   rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Apply a sequence of operators to an initial state, checking groupoid validity.

    Parameters
    ----------
    operators : [T_1, T_2, ..., T_k]  (applied left to right: T_1 first)
    initial_state : (n·d,) initial market state

    Returns
    -------
    final_state : market state after all events
    log : list of descriptions of each step
    """
    rng = rng or np.random.default_rng()
    state = initial_state.copy()
    log = []

    for i, op in enumerate(operators):
        expected_in = op.source_dim
        actual_in = len(state) // 5  # d=5 assumed

        if expected_in > 0 and actual_in != expected_in:
            raise ValueError(
                f"Event {i} '{op.name}': state has {actual_in} assets "
                f"but operator expects {expected_in}."
            )

        state = op.apply(state, rng=rng)
        log.append(
            f"  [{i:02d}] {op.name}: {expected_in}→{op.target_dim} assets  "
            f"| state L2-norm: {np.linalg.norm(state):.4f}"
        )

    return state, log
