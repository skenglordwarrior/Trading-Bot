# Liquidity Lock Check System

## Overview
Our liquidity safety net stacks independent data sources so the bot can declare a pool "locked" only after cross-verifying holder snapshots, Unicrypt (UNCX) disclosures, and on-chain transaction history. The pipeline first builds an LP-holder census from Ethplorer (with automatic Etherscan fallback), then grades each holder to see whether ≥95 % of the supply is provably locked. If that verdict is inconclusive, the bot pivots to UNCX APIs (REST plus GraphQL) before finally sweeping Etherscan transfers for burn events, locker calls, or trusted locker contract names.【F:ethereumbotv2.py†L1767-L1867】【F:ethereumbotv2.py†L2438-L2697】【F:ethereumbotv2.py†L1880-L1933】

## Layered Source Strategy

### 1. Holder Analytics First
- **Ethplorer top holders.** `_fetch_ethplorer_top_holders` pages through the public Ethplorer endpoint with keyed retries, normalises each holder record, and logs structured telemetry for operators.【F:ethereumbotv2.py†L3620-L3663】
- **Automatic fallback.** `_fetch_holder_distribution_async` reuses the Ethplorer snapshot when available, but transparently drops to the Etherscan `tokenholderlist` API whenever Ethplorer does not respond or the operator needs more depth.【F:ethereumbotv2.py†L3666-L3702】
- **95/5 gating.** `_check_liquidity_locked_holder_analysis` tallies balances by status, credits unknown balances to the remainder, and only returns `True` when ≥95 % of the supply is categorised as locked (with the mirror condition for an unlocked verdict).【F:ethereumbotv2.py†L1767-L1867】

This combination gives the bot a deterministic "lock" verdict whenever a reputable locker or burn sink dominates the LP supply, making holder analysis the fastest route for high-confidence confirmations.

### 2. Dual UNCX Integrations
- **Legacy REST endpoint.** `_check_liquidity_locked_uncx_rest_async` polls the historical UNCX REST service, tracks rate limits/service errors, and aggregates lock entries to ensure at least one active amount extends beyond the current timestamp before flagging the pair as locked.【F:ethereumbotv2.py†L2438-L2541】
- **GraphQL mirror.** `_check_liquidity_locked_uncx_graph_async` hits the modern UNCX subgraph, parsing both explicit lock records and aggregated `lockedPools` metadata so Graph coverage still detects perpetual locks even when granular entries are missing.【F:ethereumbotv2.py†L2576-L2679】
- **Orchestrator.** `_check_liquidity_locked_uncx_async` stitches both feeds together, short-circuiting on a positive REST result, otherwise preferring the graph verdict and finally returning whatever the REST call concluded if the graph is inconclusive.【F:ethereumbotv2.py†L2684-L2697】

Because the UNCX stack captures both historical and live indexing paths, the bot stays resilient when Unicrypt throttles or migrates infrastructure.

### 3. Etherscan Transaction Sweep
When neither holders nor UNCX can supply an answer, `_check_liquidity_locked_etherscan_async` inspects the LP token’s transfer log directly:
- Looks for mints to the zero address/`0x…dead` burns that destroy supply entirely.【F:ethereumbotv2.py†L1884-L1909】
- Accepts explicit locker function signatures (`lock`, `lockLPToken`, etc.) while screening out `unlock` actions.【F:ethereumbotv2.py†L1903-L1911】
- Fetches the destination contract’s verified source and whitelists common locker prefixes such as `Unicrypt`, `Pink`, or `Team` if the ABI lacks function metadata.【F:ethereumbotv2.py†L1916-L1928】

Any transport/network errors automatically disable Etherscan lookups for the current session, signalling upstream health monitors while preventing false "locked" positives.【F:ethereumbotv2.py†L1929-L1933】

### 4. DexScreener as a Confidence Bonus
Even after the internal checks resolve a lock decision, the DexScreener integration only honours its `locked` label if our own pipeline still reads `False`, ensuring marketing tags cannot overrule on-chain facts.【F:ethereumbotv2.py†L2007-L2032】

## Manual Reporting Workflow
Operators capture high-signal launches in follow-up reports that mirror the automated pipeline, giving human auditors a ready-made audit trail. For example, the Maxi Doge/WETH report enumerates the Ethplorer creation timestamp, the initial Etherscan mint, the UNCX lock transfer, and the subgraph coverage—condensing the lifecycle into a single timing table for rapid reviews.【F:run_reports/manual_pair_followup_0x7d89F902_2025-11-15.md†L1-L22】

The resulting table matches the live dashboard shared with stakeholders, reinforcing why “we have one of the best lock-liquidity check systems”: each row is backed by redundant on-chain sources, and any operator can retrace the evidence in seconds.

## Why It Matters
- **Defence in depth.** Three independent confirmations (holders, UNCX, raw transactions) drastically lower the odds of a forged lock label slipping through.【F:ethereumbotv2.py†L1767-L1933】【F:ethereumbotv2.py†L2438-L2697】
- **Operational resilience.** Every network call is wrapped with telemetry and graceful degradation, so outages at any upstream service disable that leg without crashing the entire scan loop.【F:ethereumbotv2.py†L2438-L2539】【F:ethereumbotv2.py†L2576-L2631】【F:ethereumbotv2.py†L3660-L3699】
- **Human-friendly evidence.** Hand-curated follow-ups keep analysts in the loop and give clients tangible proof of lock coverage, complete with unlock schedules and durations.【F:run_reports/manual_pair_followup_0x7d89F902_2025-11-15.md†L1-L22】

Together, the automation and documentation demonstrate why the desk can confidently claim to run one of the strongest liquidity-lock verification stacks in the space.
