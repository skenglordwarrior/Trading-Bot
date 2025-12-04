# Building a Maestro-Style Telegram Trading Bot (Without Maestro APIs)

This document outlines how to extend the existing Telegram bot to place on-chain buy/sell orders directly from inline buttons, connect trades to a local wallet, and show live profit & loss (PnL) updates. The approach avoids Maestro entirely and relies on your own transaction builder and a DEX router (e.g., Uniswap v2/v3 or 1inch) plus an indexer/price source.

## High-Level Architecture

1. **Wallet + Signing**
   - Use a hot wallet (private key or JSON keystore) loaded from environment or secure file.
   - Connect to an Ethereum (or compatible) RPC endpoint.
   - Build and sign transactions locally; broadcast via the RPC.

2. **DEX Interaction Layer**
   - Implement minimal swap helpers for the target DEX:
     - Uniswap v2: `swapExactETHForTokens`, `swapExactTokensForETH` (or WETH) through a canonical router.
     - Uniswap v3: use a quoter for price estimation and a `multicall`/`exactInputSingle` swap.
     - 1inch/0x: call their quote API, then submit the returned calldata and target address.
   - Include slippage, gas limit, priority fee, and deadline handling.

3. **Price + PnL Data Sources**
   - Pull mid-price from the DEX pool reserves or an aggregator API (DexScreener, CoinGecko, 1inch quote endpoint).
   - Cache prices per token and refresh on a timer (aligned with your existing auto-refresh loop).
   - Track token balances via on-chain calls or by parsing your own transaction receipts.

4. **Position & PnL Store**
   - Maintain a small state object keyed by token address:
     ```python
     positions[token] = {
       "qty": Decimal,           # token units held
       "avg_cost_eth": Decimal,  # cost basis in ETH
       "realized_eth": Decimal,  # PnL realized across sells
       "last_price_eth": Decimal,
       "unrealized_eth": Decimal,
       "pnl_pct": Decimal,
     }
     ```
   - Update on each confirmed buy/sell, and recompute `last_price_eth`, `unrealized_eth`, and `pnl_pct` during periodic refreshes.

5. **Telegram UX**
   - Attach inline buttons to each tracked pair message:
     - Buy presets: `Buy 0.01 / 0.05 / 0.1 ETH`.
     - Sell presets: `Sell 25% / 50% / 100%` (amount of current balance).
     - Refresh: triggers a manual price/balance refresh.
   - Handle callback queries by dispatching to the trading engine (asynchronous job), then edit the Telegram message with submission status and PnL once mined.

6. **Execution Engine**
   - Queue trades to avoid nonce collisions: a simple in-memory queue or per-token mutex is sufficient for single-user flows.
   - Steps per order:
     1. Validate slippage / balance / allowance (approve token if selling ERC-20).
     2. Build swap calldata and estimate gas.
     3. Sign transaction with the loaded wallet.
     4. Broadcast and await confirmation (or show TX hash immediately and poll in background).
     5. Update the position store and edit the Telegram message.

7. **Risk & Safety Controls**
   - Dry-run mode: skip signing and only show the would-be payload and quoted price.
   - Max spend per trade and per time window; min liquidity filters.
   - Rug/lock checks before buys; blocklist tokens; enforce max slippage.

## Implementation Steps in This Codebase

1. **Wallet + RPC wiring** (`ethereumbotv2.py` or a new `trading_engine.py`):
   - Load private key from env (e.g., `TRADER_PRIVKEY`) and create a Web3 account object.
   - Add helpers `get_nonce()`, `sign_and_send(tx)` returning TX hash/receipt.

2. **DEX swap helpers** (new module `dex_client.py`):
   - Implement `quote_buy(token_address, eth_amount)` and `quote_sell(token_address, token_amount)` using pool reserves.
   - Implement `build_buy_tx(token_address, eth_amount, slippage_bps)` that returns `to`, `data`, `value`, `gas`, `maxFeePerGas`, `maxPriorityFeePerGas`.
   - Implement `build_sell_tx(token_address, token_amount, slippage_bps)` plus an `ensure_allowance()` helper for ERC-20 approvals.

3. **Position tracker** (new module `positions.py`):
   - Store balances, avg cost, realized PnL.
   - On buy: increase qty and update weighted avg cost.
   - On sell: reduce qty, compute realized PnL, adjust avg cost.
   - Periodically update `last_price_eth` from the quote helper and compute unrealized PnL.

4. **Telegram buttons + callbacks** (`ethereumbotv2.py`):
   - Extend inline keyboards to include buy/sell presets.
   - Add callback handlers that parse payloads into actions (`buy:0.05`, `sell:50`).
   - Kick off async tasks to call `execute_buy`/`execute_sell` in the trading engine.
   - After confirmation, edit messages to show:
     - Current balance for the token.
     - Avg entry vs. last price.
     - Unrealized and realized PnL (absolute and %).
     - Last TX hash/link.

5. **Background refresh loop**
   - On the existing auto-refresh interval, update prices for all open positions and re-render messages with up-to-date PnL.
   - Include a manual refresh button for immediate updates.

6. **Testing + Safety**
   - Unit-test PnL math with fixtures for buy/sell sequences.
   - Dry-run tests for transaction builders (assert calldata/values without sending).
   - Test Telegram callback parsing to ensure correct size selection and error handling.

## Minimal Data Flows

**Buy button → on-chain trade**
1. User taps `Buy 0.05 ETH`.
2. Callback handler parses amount → calls `execute_buy(token, 0.05 ETH)`.
3. `execute_buy` quotes price, builds tx, signs, broadcasts, and stores the pending TX hash.
4. Receipt handler updates balances/positions, then Telegram message is edited with PnL and TX link.

**Sell button → on-chain trade** follows the same path with `execute_sell`, including allowance checks.

## Configuration Suggestions
- `TRADER_PRIVKEY` for the wallet; optional `WSS_RPC_URL`/`HTTP_RPC_URL`.
- `DEFAULT_BUY_AMOUNTS = [0.01, 0.05, 0.1]` (ETH) and `DEFAULT_SELL_STEPS = [25, 50, 100]` (% of balance).
- `MAX_SLIPPAGE_BPS`, `PRIORITY_FEE_GWEI`, `DEADLINE_SECONDS`, `DRY_RUN` boolean.

## What This Delivers
- Inline Telegram buys/sells directly from your wallet.
- Live PnL in each pair’s message, refreshed automatically.
- No dependency on Maestro; everything is built with standard Web3 and DEX contracts.
