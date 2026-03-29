# autotrading

This is an experiment to have the LLM autonomously research and optimize trading strategies on Bitcoin and commodities via HyperLiquid.

**MODE: PAPER TRADING ONLY.** All evaluation is done via backtesting on historical data. No real funds are ever at risk during the optimization loop. The system operates exclusively in paper mode.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar29`). The branch `autotrading/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autotrading/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data download, feature engineering, backtesting engine, evaluation. **Do not modify.**
   - `strategy.py` — the file you modify. Trading strategy logic, parameters, signal generation.
4. **Verify data exists**: Check that `~/.cache/autotrading/` contains data parquets. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs a backtest on historical data. The strategy is evaluated on the **last 90 days** (out-of-sample validation set). You launch it simply as: `uv run strategy.py`.

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit. Everything is fair game: parameters, signal logic, indicators, ML models, ensemble methods, position sizing, risk management.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, backtesting engine, feature engineering, and data loading.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_strategy` function in `prepare.py` is the ground truth metric.
- Introduce look-ahead bias. Your signals must only use data available at or before the current timestamp.

**The goal is simple: get the highest composite_score.** The composite score is:

```
composite_score = sharpe_ratio × (1 - max_drawdown) × sign(total_return)
```

This rewards strategies that:
- Have high risk-adjusted returns (Sharpe)
- Avoid large drawdowns
- Are actually profitable (positive total return)

**Minimum trades**: Strategies with fewer than 10 trades in the evaluation period score 0 (insufficient signal confidence).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the strategy as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
composite_score:  1.234567
sharpe_ratio:     1.85
sortino_ratio:    2.12
total_return:     0.3420
annualized_return:1.4500
max_drawdown:     0.1280
calmar_ratio:     11.33
win_rate:         0.583
profit_factor:    1.8500
num_trades:       156
exposure_pct:     65.3
eval_seconds:     12.5
```

You can extract the key metric from the log file:

```
grep "^composite_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	composite_score	sharpe	max_dd	status	description
```

1. git commit hash (short, 7 chars)
2. composite_score achieved (e.g. 1.234567) — use 0.000000 for crashes
3. max drawdown as decimal (e.g. 0.128)
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	composite_score	sharpe	max_dd	status	description
a1b2c3d	0.850000	1.20	0.150	keep	baseline
b2c3d4e	1.120000	1.65	0.120	keep	increase fast MA to 30, add vol scaling
c3d4e5f	0.430000	0.80	0.250	discard	switch to pure mean reversion
d4e5f6g	0.000000	0.00	0.000	crash	LSTM model OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autotrading/mar29`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `strategy.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run strategy.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^composite_score:\|^sharpe_ratio:\|^max_drawdown:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If composite_score improved (higher), you "advance" the branch, keeping the git commit
9. If composite_score is equal or worse, you git reset back to where you started

## Trading-specific guardrails

**No look-ahead bias**: This is the #1 sin in backtesting. Your strategy must NEVER use future information to make decisions. The `prepare.py` evaluation harness applies a 1-bar delay (signal at close → trade at next open), but your signal logic itself must only reference past data. If you see suspiciously perfect results (Sharpe > 5, 90%+ win rate), you probably have look-ahead bias.

**Transaction costs are real**: The backtester deducts 3.5 bps commission + 2.0 bps slippage per trade. Strategies that trade every bar will be destroyed by costs. Think in terms of meaningful position changes.

**Funding costs matter**: On HyperLiquid, long positions pay positive funding rates and short positions receive them (when funding is positive, reversed when negative). The backtester accounts for this. Strategies that hold large long positions during high funding periods will underperform.

**Overfitting warning**: The train/val split is temporal (first 275 days train, last 90 days val). If your strategy works perfectly on training data but fails on validation, it is overfit. Be especially wary of strategies with many parameters tuned to specific patterns.

**Position limits**: Signals must stay in [-1, +1]. The backtester applies max leverage of 3x. A signal of 1.0 means "go max long with 3x leverage".

## Idea categories to explore

1. **Parameter tuning**: MA lengths, RSI thresholds, position sizing, vol targets
2. **Signal combination**: Ensemble multiple indicators with weighting
3. **Regime detection**: Trending vs mean-reverting markets (use volatility, ADX, etc.)
4. **Volatility-adjusted sizing**: Risk parity, inverse-vol weighting
5. **Multi-asset correlation**: BTC leads, alts follow; correlation-based hedging
6. **Funding rate signals**: Crowded positioning → contrarian opportunities
7. **ML approaches**: Small LSTM/GRU on features, gradient boosted trees, simple neural nets
8. **Risk management**: Dynamic stops, trailing stops, max drawdown circuit breaker
9. **Time-based patterns**: Hour-of-day, day-of-week effects
10. **Breakout strategies**: Donchian channel, volatility breakout
11. **Multi-timeframe**: Combine signals from different lookback windows

## NEVER STOP

Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each backtest takes seconds, so you can run dozens of experiments per hour. The user then wakes up to a full history of experiments, all completed by you while they slept!
