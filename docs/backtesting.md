# Backtesting pipeline

The bot now records a structured snapshot every time a pair clears the 14-point
checklist so you can replay outcomes later. Snapshots are stored as JSON lines
under `backtests/passing_snapshots.jsonl` with the full DexScreener payload,
check breakdown, and contract risk metadata. This lets you correlate PnL with
specific gates (e.g., unrenounced contracts vs. fully renounced with locked
liquidity).

## Snapshot contents
Each entry captures:

- Pair address, token address, symbol/name
- Timestamp, passes/total counts, check breakdown
- DexScreener metrics (price, liquidity, FDV/MC, buys/sells, lock status)
- Contract hygiene (risk score, renounce flag)
- The raw checklist payload for future feature engineering

Snapshots are appended automatically when a pair first hits passing status,
when a recheck moves it into passing, and when a passing-refresh keeps it in
passing.

## Running backtests
Use the CLI helper to replay snapshots against DexScreener TradingView candles
and compute PnL multiples, drawdowns, and optional stop/take hits:

```bash
python -m backtesting --pair 0xYourPair --horizon 360 --resolution 5 --take-profit 3 --stop-loss 0.7
```

Key flags:
- `--pair`: Restrict to a specific pair (omit to backtest all snapshots)
- `--limit`: Only backtest the most recent N snapshots
- `--horizon`: Holding period in minutes (default 240)
- `--resolution`: DexScreener TradingView resolution (default 15m)
- `--take-profit`: Exit multiple vs. entry close if reached intrabar
- `--stop-loss`: Stop multiple vs. entry close if reached intrabar

CLI output summarizes each run as `pair | pnl xM | peak xM | drawdown D% | bars`.

## Filesystem layout
- `backtesting.py`: Backtest engine and CLI entrypoint
- `backtests/passing_snapshots.jsonl`: Append-only snapshot log
