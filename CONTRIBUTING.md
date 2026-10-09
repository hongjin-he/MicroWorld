# Contributing to MicroWorld

Thank you for your interest. MicroWorld is a research-grade framework — contributions are welcome across theory, implementation, and empirical validation.

## Quick Start for First-Time Contributors

New here? Start with an issue labelled [`good first issue`](https://github.com/hongjin-he/MicroWorld/labels/good%20first%20issue) — each one is scoped to a few hours and doesn't require reading the theory.

```bash
git clone https://github.com/<you>/MicroWorld && cd MicroWorld
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
pytest -q            # should be all green before you change anything
```

Comment on the issue to claim it, open a PR that says `Closes #<n>`, and make sure `pytest -q` passes. Small PRs are merged fastest. Issues-first is only required for the larger theory/implementation changes described below.

---

## Ways to Contribute

### Theory Extensions

The framework has natural open problems we actively want external perspectives on:

- **Level 0 (cross-market):** Empirical calibration of cross-market coupling functionals $\Psi^{(1\to0)}$ using North-South capital flow data (A-share 北向资金, EM flows)
- **Information architecture:** Estimating type-level SNR differentials using alternative data APIs; formalizing how information asymmetry updates during macro shocks
- **Event algebra:** Identifying edge cases where the groupoid composition law breaks (e.g., simultaneous merger + IPO in the same sector)
- **Lévy calibration:** Fitting $\nu^\eta$ to intraday order flow data for different institution-type signatures

### Empirical Validation

We have one documented Lyapunov signal (COVID-19 pre-crash, Feb 20 2020, RI = 0.83). We need:

- Additional retrospective runs on 2008, LTCM, 2018 Q4, Flash Crash
- Cross-market validation: does Level 0 coupling capture A-share 熔断 dynamics (Jan 2016)?
- Comparison of RI$(t)$ lead time across crisis types (liquidity vs. solvency vs. contagion)

### Implementation (Engineering Repo)

The engineering code lives in this repo (`agents/`, `events/`, `game/`, `state/`, `controller/`, `backtest/`). Good places to contribute:

- Level 0 cross-market module (current implementation covers Levels 1–3 only)
- Alternative data connectors for information hierarchy calibration
- Performance optimization for the neural fictitious play loop

---

## Contribution Process

1. **Open an issue first** for any non-trivial change — especially theory extensions that touch the core 8-equation HJB-FPK system. This prevents duplicate work and lets us flag potential conflicts with in-progress work.

2. **Fork and branch** — use a descriptive branch name:
   ```
   theory/level0-empirical-calibration
   empirics/2008-crisis-retrospective
   impl/levy-calibration-fitting
   ```

3. **Mathematical contributions** should include:
   - Theorem statements with explicit hypothesis conditions
   - Proof sketches (full proofs in appendix or separate LaTeX file)
   - Connection to existing components (cite theorem numbers from main README)

4. **Empirical contributions** should include:
   - Data source and vintage
   - Replication script (Python, reproducible environment via `requirements.txt`)
   - Results table with confidence intervals

5. **Open a Pull Request** against `main`. The PR description should state:
   - What problem this addresses
   - How it connects to the existing theoretical framework
   - Any open questions you'd like reviewers to focus on

---

## Code Standards (Engineering Contributions)

- Python 3.10+, type annotations throughout
- Tests via `pytest`; new modules require at least one integration test
- No silent dependencies on proprietary data — all demos must run on public data or synthetic data
- Document the mathematical object each class implements (HJB solver, FPK integrator, etc.)

---

## Discussion

For theoretical discussion before opening a formal issue, use GitHub Discussions. For time-sensitive research collaboration inquiries, reach out via [LinkedIn](https://www.linkedin.com/in/hongjinhe-hkust-edu).

---

## Code of Conduct

This is a research repository. Contributions should be substantive, honest about limitations, and respectful of others' work. We hold ourselves to the same standard — if we cite your work incorrectly or misrepresent a comparison, please open an issue and we will correct it promptly.
