# Changes

Branch: `fix/dual-noise-units-and-compose-cholesky`
Commit: `2265b84`

Two bugs fixed. Both were about **units and matrix algebra**, not about logic
or features. Nothing was added or removed — two calculations were wrong and are
now right.

---

## Bug 1 — the jump counter wasn't a rate

**File:** `state/noise.py`

**What it should do:** measure how often a market jumps, in *jumps per day*.

**What it did:** multiplied the jump count by the bar width instead of dividing
by elapsed time. That's the upside-down version of a rate.

```python
lambda_eta = jump_mask.sum() * dt      # before — a count, scaled wrongly
lambda_eta = jump_mask.sum() / (n*dt)  # after  — a rate
```

The volatility estimate `sigma_tau` had the same problem: it measured
volatility *per bar* instead of *per day*.

**Why it mattered:** the repo's headline result (Theorem III.1, the
Cramér-Rao bound) **adds** these two numbers together. If one is per-bar and
the other is per-day, adding them is meaningless — like adding 5 kilometres to
3 hours.

**Measured effect.** A test market with a known 3 jumps/day and 2% daily
volatility:

| | true | before | after |
|---|---|---|---|
| jumps per day | 3.0 | 0.38 | **3.00** |
| daily volatility | 0.020 | 0.0022 | **0.0198** |

Worse than being wrong, it was *inconsistently* wrong: feeding the same market
in as 5-minute bars vs 1-minute bars gave answers 5x apart. The Cramér-Rao
bound is supposed to describe a market, not your data feed. It now agrees
within 1% across both.

**One caller fixed:** `demo/run_egamec.py` passed `dt=1/252` for daily bars,
which told the code each bar was 1/252 of a day. Corrected to `dt=1.0`.

---

## Bug 2 — event composition crashed on market crises

**File:** `events/operators.py`

**What it should do:** combine two market events into one (e.g. "rate hike
after a systemic crisis") and work out the combined uncertainty.

**What it did:** `Sigma_w` holds noise in a form where the real covariance is
`S x S-transpose`. Every other part of the codebase reads it that way.
`compose()` used `S x S` — no transpose.

```python
op1.Sigma_w @ op1.Sigma_w      # before
op1.Sigma_w @ op1.Sigma_w.T    # after
```

**Why it mattered:** the two forms only agree when the matrix is diagonal.
Exactly one event type has a non-diagonal one — `systemic_crisis_operator`,
where the whole point is that asset prices fall *together*.

For that operator the wrong form produces a matrix that isn't a valid
covariance at all, so the next line crashed:

```
LinAlgError: Matrix is not positive definite
```

**In plain terms: you could not combine a market crisis with any other event.**
Not "you got a wrong number" — the code stopped.

Nothing in the codebase composed a crisis operator, and every existing
composition test used diagonal-noise events, so this was never hit.

---

## Files changed

| File | What |
|---|---|
| `state/noise.py` | 2 lines of math + docstrings stating the time units |
| `events/operators.py` | 1 line of math + a comment on why |
| `demo/run_egamec.py` | 1 line — corrected `dt` for daily bars |
| `tests/test_noise.py` | +6 tests (new, additive) |
| `tests/test_events.py` | +5 tests (new, additive) |

Total: 268 added, 9 removed. Only **4 lines** are real logic; the rest is
tests and comments.

---

## Testing

Suite: **133 -> 144 tests, all passing.**

Each new test was checked against the old code to confirm it actually catches
the bug: **5 of 6** new noise tests and **4 of 5** new event tests fail on the
previous commit. The ones that pass on old code are intentional — each group
keeps one diagonal-noise case to make sure previously-correct behaviour didn't
move.

The tests check *meaning*, not memorised numbers. The strongest one holds the
jump **count** fixed while stretching the time window — 6 jumps in 2 days is a
different rate than 6 jumps in 20 days. A count-shaped estimator returns the
same number for both and gets caught.

---

## What still works

- All 144 tests pass
- `demo/run_egamec.py`, `demo/denoised_price_2026.py`, `demo/hindcast_2008.py`
  all run clean
- CI is unaffected (it runs `pytest tests/` plus 3 import checks)
- `compose()` has zero non-test callers, so that fix can't regress anything —
  it only makes a previously-impossible operation possible

## What this affects downstream

`notebooks/day03_dual_noise.ipynb` prints the estimate next to a hardcoded
"true" value. That line now visibly disagrees.

The estimate is the part that's now **correct**: it returns 0.190 against that
notebook's true annual volatility of 0.20 (its `dt` is in years, so the answer
comes out annualised). The stale part is the hardcoded string, which is a
per-bar number labelled "/day".

Two-line notebook fix. Not included in this commit.

---
