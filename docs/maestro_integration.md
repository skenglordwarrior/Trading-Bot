# Maestro integration guide

This repository's Ethereum bot is currently a research and monitoring tool. It scores new liquidity pools with `check_pair_criteria` and then schedules volume checks or refreshes, but it does not dispatch any swap/buy transactions.

## Where to hook a buy trigger
- **Pair evaluation:** `check_pair_criteria` returns `passes`, `total`, and a rich `extra` payload with DexScreener stats, contract risk, and verification status. It is already used as the single gate before any follow-up work.
- **Lifecycle pivot:** `handle_new_pair` handles the initial pass/fail outcome. Once `passes >= MIN_PASS_THRESHOLD`, it starts wallet monitoring and queues volume checks or passing refresh timers. This is the earliest point where a trade trigger would be safe to add without duplicating work or missing requeues.

Recommended insertion point:
1. Inside `handle_new_pair`, immediately after the `passes >= MIN_PASS_THRESHOLD` block that already increments metrics and calls `start_wallet_monitor(main_token)`.
2. Gate the call on both the pass threshold and the volume/trade guard (`vol_now >= MIN_VOLUME_USD` and `trades_now >= MIN_TRADES_REQUIRED`) to avoid frontrunning illiquid pools.

## Sketch of an execution hook
1. **Config:** Add Maestro API URL, API key/token, and a default size/slippage to the existing environment-driven config (align with `INFURA_URL` and similar settings).
2. **Client:** Create a small `maestro_client.py` module that exposes `submit_buy(pair_address, base_token, quote_token, amount)` and handles retries + 429 back-off.
3. **Trigger:** Call `submit_buy` from the recommended insertion point with the pair address, token addresses, and desired size. Make the call async-safe (use `asyncio.run` or `loop.create_task` if a long request) so it does not block the main scanning loop.
4. **Result handling:** Log Maestro responses alongside the existing `log_event` metadata, and push success/failure to Telegram so operators can reconcile fills.

## Safety considerations
- Keep the existing pass threshold, volume/trade minimums, and verification error handling intact so risky pools are not traded.
- Fail closed: if the Maestro call errors, log and continue monitoring rather than retrying infinitely in the hot path.
- Store secrets in env vars or a separate, git-ignored file; do not hardcode API tokens.
- Add a dry-run flag so you can shadow-test the hook before enabling live orders.
