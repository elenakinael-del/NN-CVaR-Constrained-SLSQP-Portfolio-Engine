Adaptation of the KNN macro-regime / Bellman utility / CVaR-SLSQP architecture to a single-instrument (COMEX Gold, GC=F) context.

Since gold is one instrument rather than a 50-stock universe, the "assets" being allocated across are a bank of systematic gold sub-strategies (trend, momentum, mean-reversion, breakout, vol-targeting, real-yield/DXY macro overlays). The engine:

Builds a macro feature matrix (VIX, DXY momentum, real-yield change, credit spread, realized vol, breadth)
Uses K-Nearest-Neighbours (K=60) with a softmax kernel to match the current macro state to historical analog regimes, producing a time-varying risk-aversion (λₜ) and CVaR penalty (κₜ)
Formulates a Bellman-style one-step utility: U(w) = w'μ − (λ/2)·w'Σw − κ·CVaR₀.₀₅(w)
Optimizes strategy weights via scipy SLSQP under long-only, bounded, fully-invested constraints
Runs a Monte Carlo sweep over random strategy subsets + regime-parameter draws to build a universe of candidate portfolios, then ranks them by realized utility
Renders the same 4-panel dashboard (risk/return, top-5 equity curves, drawdown distribution, utility-vs-Sharpe) against a buy-and-hold gold benchmark
