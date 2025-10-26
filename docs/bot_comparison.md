# Bot Comparison: `telegrambot.py` vs. `ethereumbotv2.py`

## Overview
- **`ethereumbotv2.py`** is the primary production bot. It layers structured queue management, richer metrics, wallet tracker integration, and advanced contract verification on top of the scanning loop. Its entry point (`main`) bootstraps persisted block heights, validates Etherscan connectivity, and then orchestrates Uniswap log ingestion, pair evaluation, rechecks, silent monitoring, and wallet updates.【F:ethereumbotv2.py†L4471-L4638】
- **`telegrambot.py`** is the streamlined legacy bot focused on human-readable Telegram alerts. Its `main` loop mirrors the scanning pattern but logs directly to the console and omits several of the auxiliary safety rails that the v2 bot includes.【F:telegrambot.py†L3568-L3666】

Both bots share the same broad workflow: connect to multiple RPC providers, watch the Uniswap factories for new pools, query DexScreener for fresh listings, analyse each candidate with on-chain heuristics (honeypot, renounce, liquidity, wallet activity), and broadcast alerts.

## Feature and Function Map
The tables below group related functionality, highlighting the functions that implement each feature in both bots. Rows list shared capabilities; bullet points underneath capture functions that exist only in one bot.

### Bootstrapping & Persistence
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| Excel initialisation | `init_excel` prepares `pairs.xlsx` for historical storage.【F:telegrambot.py†L270-L304】 | Same helper reused for the v2 bot to ensure identical reporting structure.【F:ethereumbotv2.py†L520-L555】 | Both create identical workbook schemas.
| Block persistence | `load_last_block`, `save_last_block` track the last processed Uniswap block for each factory.【F:telegrambot.py†L3551-L3566】 | Same helpers, but v2 calls `ensure_etherscan_connectivity` before entering the loop to guarantee contract checks are available.【F:ethereumbotv2.py†L4455-L4494】【F:ethereumbotv2.py†L4471-L4494】 | v2 adds a proactive Etherscan validation step.

### Logging, Metrics, and Notifications
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| Console logging | `logging.basicConfig` emits `[LEVEL] timestamp - message` output for readability.【F:telegrambot.py†L252-L276】 | `logging.basicConfig` now mirrors the Telegram formatter, while `log_event` adds action/pair/context metadata in-line.【F:ethereumbotv2.py†L323-L360】 | The v2 bot’s output now matches Telegram’s clarity per the new requirement.
| Telegram delivery | `send_telegram_message`, `send_telegram_json` wrap the Telegram Bot API, along with formatting helpers (`send_ui_criteria_message`, `send_bull_insights`).【F:telegrambot.py†L930-L1112】【F:telegrambot.py†L2666-L2751】 | Identical messaging helpers ensure parity; v2 reuses them for automated alerting during queue processing.【F:ethereumbotv2.py†L900-L1084】【F:ethereumbotv2.py†L3110-L3206】 | Function sets are shared, preserving message structure across bots.
| Metrics | Basic console counters only (no dedicated class). | `MetricsCollector` aggregates per-minute statistics, RPC/API latencies, and emits structured alerts for slow throughput or error spikes.【F:ethereumbotv2.py†L420-L515】 | Metrics and health monitoring are unique to v2.

### Web3 Connectivity & Event Handling
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| Provider bootstrap | Instantiates primary and backup `Web3` providers for Uniswap v2/v3 and read RPCs.【F:telegrambot.py†L308-L350】 | Same pattern, plus provider rotation helpers (`_rotate_rpc_provider`, `safe_block_number`, `safe_get_logs`) to automatically fall back when an endpoint fails.【F:ethereumbotv2.py†L210-L403】 | v2 adds automatic rotation with metrics instrumentation.
| Wallet tracker integration | `wallet_tracker` import fallback ensures notifier callbacks are wired for wallet activity alerts.【F:telegrambot.py†L86-L127】 | Shares the same tracker hooks but also exposes `_etherscan_get_async` to the tracker and toggles lookups when rate limits occur.【F:ethereumbotv2.py†L60-L140】【F:ethereumbotv2.py†L664-L704】 | v2 keeps the tracker informed about Etherscan availability.

### DexScreener and Market Data
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| DexScreener client | `_fetch_dexscreener_data_async` / `fetch_dexscreener_data` cache responses, attach lock-status, and aggregate token metadata.【F:telegrambot.py†L1148-L1242】 | Same interface, reused by the v2 criteria engine and gem detection modules.【F:ethereumbotv2.py†L1096-L1258】 | Shared implementation keeps market data consistent.
| Volume & milestone tracking | `queue_volume_check`, `handle_volume_checks`, `check_marketcap_milestones` gate promotions until liquidity/volume thresholds are met.【F:telegrambot.py†L2802-L2962】 | v2 mirrors these mechanics for automated refreshes and silent checks.【F:ethereumbotv2.py†L3408-L3612】 | Behaviour aligned across bots.

### Honeypot, Renounce, and Wallet Safety Checks
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| Honeypot.is integration | `check_honeypot_is` and quick re-check timers guard against toggled honeypots.【F:telegrambot.py†L1460-L1546】 | Identical functions reused during initial screening and refresh loops.【F:ethereumbotv2.py†L1606-L1688】 | Shared protection layer.
| Ownership & renounce | `check_is_renounced`, `check_renounced_by_event`, and owner balance queries feed risk scoring and notifications.【F:telegrambot.py†L1388-L1454】【F:telegrambot.py†L1548-L1650】 | Same helpers plus tighter integration with contract verification flags and wallet analytics.【F:ethereumbotv2.py†L1480-L1587】【F:ethereumbotv2.py†L1855-L1894】 | v2 threads the results through advanced scoring.
| Wallet analytics | `get_wallet_report`, `handle_wallet_updates`, `analyze_seller_wallet` provide marketing/risk scoring and smart-money alerts.【F:telegrambot.py†L2140-L2489】 | v2 reuses these analytics inside passing refresh logic and gem detectors.【F:ethereumbotv2.py†L2182-L2542】 | Shared wallet intelligence across bots.

### Etherscan Contract Intelligence
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| API configuration | Static API URL list with key rotation helpers.【F:telegrambot.py†L1012-L1106】 | Enhanced helpers add `_prepare_etherscan_params`, chain-id injection, endpoint validation (`ensure_etherscan_connectivity`), and dynamic disabling when failures occur.【F:ethereumbotv2.py†L1208-L1372】 | v2 now fully supports the Etherscan v2 API and surfaces clearer diagnostics.
| Source retrieval | `_fetch_contract_source_etherscan_async` fetches verified source blobs and parses multi-file layouts.【F:telegrambot.py†L1252-L1338】 | Same logic augmented with the new chain-id parameter and graceful degradation when lookups are disabled.【F:ethereumbotv2.py†L1386-L1499】 | Fix resolves the earlier “risk 9999” issue.
| Advanced contract check | `advanced_contract_check` combines proxy detection, source analysis, Slither output, owner behaviour, and private-sale heuristics into a composite risk score.【F:telegrambot.py†L1287-L1409】【F:telegrambot.py†L2003-L2106】 | v2 extends the same routine, feeding results into `check_pair_criteria`, recheck pruning, and passing refresh watchdogs.【F:ethereumbotv2.py†L1952-L2090】【F:ethereumbotv2.py†L3223-L3360】 | Shared core, but v2 more tightly couples contract risk with lifecycle management.
| Holder & transfer analytics | Holder distribution, transfer history, private-sale detectors mine Etherscan data for concentration red flags.【F:telegrambot.py†L1401-L1715】【F:telegrambot.py†L2003-L2119】 | v2 reuses the detectors and now logs the full API `result` payload whenever Etherscan returns `NOTOK`, aiding diagnostics if rate limits recur.【F:ethereumbotv2.py†L1952-L2169】【F:ethereumbotv2.py†L628-L704】 | Additional context makes troubleshooting easier when checks fail.

### Pair Lifecycle Management
| Area | `telegrambot.py` | `ethereumbotv2.py` | Notes |
| --- | --- | --- | --- |
| Initial evaluation | `check_pair_criteria` scores new pools using DexScreener stats, contract risk, honeypot results, renounce status, and wallet analytics.【F:telegrambot.py†L2582-L2859】 | Same scoring with identical thresholds, feeding queueing logic and fail reasons used for telemetry and removals.【F:ethereumbotv2.py†L3223-L3440】 | Shared logic ensures both bots judge pairs consistently.
| Queue & retry handling | `queue_recheck`, `handle_rechecks`, `maybe_flush_failed_rechecks` manage transient DexScreener failures and enforce retry limits.【F:telegrambot.py†L2964-L3126】 | v2 mirrors this queue plus adds `handle_passing_refreshes`, `handle_volume_checks`, and silent monitoring for MC milestones.【F:ethereumbotv2.py†L3442-L3754】 | v2 emphasises automated lifecycle actions for passing pairs.
| New pair ingestion | `handle_new_pair` wraps pair evaluation, failure logging, and Excel persistence, while also detecting marketing milestones and bull-market indicators.【F:telegrambot.py†L3128-L3476】 | v2 adds structured `log_event` emissions, fail buffers, and integration with enhanced detection modules before looping back to the scheduler.【F:ethereumbotv2.py†L4139-L4426】 | Additional telemetry and removal rules help the production bot stay stable.

### Unique Enhancements in `ethereumbotv2.py`
- **Metrics-driven watchdogs**: `MetricsCollector.emit_metrics` and alert helpers detect stalls, API error spikes, and queue depth issues.【F:ethereumbotv2.py†L448-L515】
- **Endpoint resilience**: `_prepare_etherscan_params` automatically adds the required `chainid` parameter, while `disable_etherscan_lookups` toggles the wallet tracker to avoid cascading errors.【F:ethereumbotv2.py†L1234-L1357】
- **Improved logging clarity**: `log_event` now matches Telegram’s concise style but preserves action/pair/context metadata for operators.【F:ethereumbotv2.py†L323-L360】

### Legacy Convenience in `telegrambot.py`
- **Simpler console output**: direct `logger.info` calls (without `log_event`) keep ad-hoc debugging straightforward.【F:telegrambot.py†L3568-L3666】
- **Human-friendly Telegram summaries**: helpers such as `format_wallet_report`, `build_passing_message`, and `send_borderline_warning` focus on easily digestible chat updates.【F:telegrambot.py†L2110-L2362】【F:telegrambot.py†L2666-L2751】

## Takeaways
- The two bots deliberately share the critical evaluation logic (DexScreener metrics, honeypot/renounce checks, contract analysis), ensuring identical scoring for new pools.
- `ethereumbotv2.py` layers operational safeguards—metrics, endpoint validation, queue automation, and enriched logging—on top of the shared core, making it more resilient for unattended production use.
- The latest Etherscan improvements (chain-id aware requests, startup validation, richer error logging) eliminate the recurring verification failure that produced `riskScore` 9999, restoring full automated contract vetting for pairs such as `0x418D…e418`.
