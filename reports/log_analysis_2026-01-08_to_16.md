# Log Analysis (2026-01-08 through 2026-01-16)

## Scope
Analyzed the nine log files `bot.log.2026-01-08` through `bot.log.2026-01-16` for wallet tracker activity, API traffic/latency, and recheck delays.

## Methods (commands used)
- `rg -n "wallet" bot.log.2026-01-08 bot.log.2026-01-09 bot.log.2026-01-10 bot.log.2026-01-11 bot.log.2026-01-12 bot.log.2026-01-13 bot.log.2026-01-14 bot.log.2026-01-15 bot.log.2026-01-16`
- `rg -n "activity" bot.log.2026-01-08 bot.log.2026-01-09 bot.log.2026-01-10 bot.log.2026-01-11 bot.log.2026-01-12 bot.log.2026-01-13 bot.log.2026-01-14 bot.log.2026-01-15 bot.log.2026-01-16`
- `rg -n "pending_rechecks": 92 bot.log.2026-01-*`
- `rg -n '"api_calls": 186' bot.log.2026-01-*`
- `rg -n "requeue_missing_dexscreener" bot.log.2026-01-09`
- `python - <<'PY' ...` (custom parsing script to tally metrics, requeues, rechecks, and API error counts across the nine log files)

## Key findings

### 1) Wallet-activity lookups are a high-traffic, high-failure source
- Wallet tracker lookups are getting disabled during Etherscan timeouts, which indicates repeated retry pressure and likely alert traffic bursts. Example: owner activity requests time out repeatedly and both general and wallet-tracker Etherscan lookups are disabled. (See `bot.log.2026-01-08`, lines 2112–2113.)
- Wallet tracker lookups are also disabled on HTTP 502 from Etherscan, showing instability during wallet monitoring. (See `bot.log.2026-01-09`, line 4778.)
- Ethplorer holder distribution calls show frequent 400 errors in the same windows as pair processing and rechecks, implying repeated failed requests per token. Example: repeated 400s during a pair scan window. (See `bot.log.2026-01-09`, lines 4755–4756.)

### 2) Requeue and recheck delays are driven by missing DexScreener listings
- The primary requeue reason across logs is `requeue_missing_dexscreener` with a `retry_window` of 1800 seconds (30 minutes). This delay is explicitly logged in requeue events. (See `bot.log.2026-01-09`, line 149.)
- Since newly discovered pairs enter a 30-minute retry window when not listed, the first meaningful recheck can occur long after initial activity, explaining late alerts (200+ buys/sells before first recheck). Example: the requeue event logs a 30-minute retry window before rechecks can proceed. (See `bot.log.2026-01-09`, line 149.)

### 3) Backlog in pending rechecks is substantial at peak times
- The pending recheck queue reaches 92 entries, indicating a large backlog that delays recheck attempts and downstream alerts. Example metrics show `pending_rechecks: 92` at multiple intervals. (See `bot.log.2026-01-15`, line 22086.)
- During the same period, API calls per minute spike, implying load increases while the backlog remains high. Example: a metrics snapshot reports `api_calls: 186`, `pending_rechecks: 52`, and `new_pairs: 29`. (See `bot.log.2026-01-14`, line 5533.)

## Recommendations

### A) Reduce wallet activity alert traffic
1. **Rate-limit and aggregate wallet alerts**: throttle owner activity lookups per wallet/token and emit a summarized alert per interval instead of per transaction. This should reduce Etherscan calls and alert spam while maintaining actionable info.
2. **Cache wallet activity results**: store recent wallet activity results and avoid re-fetching within a short TTL (e.g., 2–5 minutes) unless a new block boundary or significant change is detected.
3. **Fail open with reduced sampling**: when Etherscan errors occur (timeouts/502s), downgrade to a sampled or delayed wallet activity check, rather than repeated immediate retries. This avoids spiraling call volume.

### B) Address recheck delays and missing DexScreener data
1. **Short-circuit rechecks for newly launched tokens**: if a token is newly discovered and DexScreener is not listed, consider a fast “initial sweep” using on-chain data before queuing a long retry window.
2. **Adaptive retry window**: use a shorter retry window (e.g., 2–5 minutes) for young pairs, then expand if the pair remains unlisted, to prevent 30-minute blind spots for fresh launches.
3. **Early high-volume heuristic**: when buy/sell counts increase rapidly, prioritize or fast-track the pair to reduce alert delays even if DexScreener is missing.

### C) Lower overall API traffic
1. **Centralized request coalescing**: coalesce repeated Etherscan/Ethplorer calls for the same token within a short window to cut duplicate traffic.
2. **Dynamic concurrency limits**: decrease API concurrency when `pending_rechecks` grows to avoid overwhelming external APIs and creating more retries.
3. **Structured error budget**: if API error rates exceed a threshold (e.g., >10% in a minute), degrade non-critical checks (wallet activity/holder distribution) to protect core pair detection.

## Institutional delay guardrails (implemented)
- **Priority rechecks for newly discovered pairs**: DexScreener-missing pairs are now queued as high-priority so they recheck sooner and more consistently during the first 10 minutes, helping reduce early-launch blind spots.
- **SLA-based scheduling**: if a recheck is overdue beyond a configured SLA window, it is surfaced for immediate processing and tracked via a `recheck_overdue` metric, giving visibility into refresh delays.
- **Backlog-aware minimum throughput**: when high-priority rechecks exist, the recheck loop now reserves a minimum per-cycle budget to prevent high-priority items from starving behind large backlogs.
- **Operational tuning knobs**: the following environment variables can be adjusted without code changes to keep delays bounded as traffic grows: `RECHECK_PRIORITY_DELAY_SECONDS`, `RECHECK_PRIORITY_WINDOW_SECONDS`, `RECHECK_SLA_OVERDUE_SECONDS`, and `RECHECK_PRIORITY_MIN_PER_CYCLE`.

## Institutional lifecycle dataset (implemented)
- **Lifecycle windows**: snapshot schedule supports early and long horizons (default 1m, 5m, 30m, 1h, 4h, 6h, 12h) via `LIFECYCLE_SNAPSHOT_OFFSETS_SECONDS` so you can model both early momentum and medium-hold outcomes.
- **Multi-source pricing**: snapshot collection pulls price data from DexScreener by default, with optional CoinGecko fallback via `LIFECYCLE_PRICE_SOURCES=dexscreener,coingecko` to widen coverage when one vendor is slow.
- **Consistent feature payloads**: each snapshot writes a structured JSONL row with price, market cap, liquidity, volume, buys/sells/trades, and source metadata to `backtests/lifecycle_snapshots.jsonl` for ML-ready ingestion.
- **Operational guardrails**: snapshot throughput is capped per maintenance cycle (`LIFECYCLE_SNAPSHOT_MAX_PER_CYCLE`) and can include raw DexScreener payloads for deeper feature engineering (`LIFECYCLE_INCLUDE_RAW_DEX=1`).
