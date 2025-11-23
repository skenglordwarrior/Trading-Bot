# Maestro integration guide

# Live Maestro execution path

The bot now includes a live Maestro execution hook. When a pair clears the production thresholds (`passes >= MIN_PASS_THRESHOLD`, volume, and trade count) **and** the pool is a WETH-paired token, the bot can dispatch a Maestro buy request without blocking the scanning loop.

**Configuration (env vars):**

- `MAESTRO_ENABLED=true` — master toggle.
- `MAESTRO_API_BASE_URL` — e.g., `https://api.maestro.xyz` (no trailing slash).
- `MAESTRO_API_KEY` — Maestro bearer token.
- `MAESTRO_ACCOUNT` — Maestro-managed account/wallet identifier.
- `MAESTRO_BUY_AMOUNT_ETH` — amount of WETH to spend per trigger (Decimal). If `0` or unset, orders are skipped.
- `MAESTRO_SLIPPAGE_BPS` — slippage in basis points (default `75`).
- `MAESTRO_PRIORITY_FEE_GWEI` — optional priority tip.
- `MAESTRO_DRY_RUN` — keep `true` to shadow-test orders; set `false` to go live.
- Optional: `MAESTRO_MAX_RETRIES`, `MAESTRO_RETRY_BACKOFF`, `MAESTRO_REQUEST_TIMEOUT`, `MAESTRO_TRADE_PATH`, `MAESTRO_BUY_REASON`.

## Where to hook a buy trigger
- **Pair evaluation:** `check_pair_criteria` returns `passes`, `total`, and a rich `extra` payload with DexScreener stats, contract risk, and verification status. It is already used as the single gate before any follow-up work.
- **Lifecycle pivot:** `handle_new_pair` handles the initial pass/fail outcome. Once `passes >= MIN_PASS_THRESHOLD`, it starts wallet monitoring and queues volume checks or passing refresh timers. This is the earliest point where a trade trigger would be safe to add without duplicating work or missing requeues.

The hook now lives inside `handle_new_pair` after the volume/trade gate. When the pair passes and meets activity requirements, `maybe_execute_maestro_buy` prepares a payload and ships it to Maestro in a background thread. It only triggers when one leg is WETH to guarantee swap routing.

## Sketch of an execution hook
1. **Config:** Provide Maestro URL, key, account, per-trade amount, and slippage via env vars.
2. **Client:** `maestro_client.MaestroClient` wraps the HTTP API with retry/back-off and rate-limit handling.
3. **Trigger:** `maybe_execute_maestro_buy` calls `MaestroClient.submit_buy_background(...)`, handing pair/token metadata and respecting dry-run mode.
4. **Result handling:** `log_event` emits structured success/failure entries and pushes Telegram failures for visibility.

## Safety considerations
- Keep the existing pass threshold, volume/trade minimums, and verification error handling intact so risky pools are not traded.
- Fail closed: if the Maestro call errors, log and continue monitoring rather than retrying infinitely in the hot path.
- Store secrets in env vars or a separate, git-ignored file; do not hardcode API tokens.
- Use `MAESTRO_DRY_RUN=true` while shadow-testing the hook before enabling live orders.
- Obtain your Maestro API key and account/wallet identifier from the Maestro dashboard or API portal. This project does **not** generate those credentials; availability and pricing are determined by Maestro (a funded Maestro account is typically required to place buys).
