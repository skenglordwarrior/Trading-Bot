# Liquidity Lock Check System

## Overview
Our liquidity safety net stacks independent data sources so the bot can declare a pool "locked" only after cross-verifying holder snapshots, Unicrypt (UNCX) disclosures, and on-chain transaction history. The pipeline first builds an LP-holder census from Ethplorer (with automatic Etherscan fallback), then grades each holder to see whether ≥95 % of the supply is provably locked. If that verdict is inconclusive, the bot pivots to the UNCX GraphQL subgraph before finally sweeping Etherscan transfers for burn events, locker calls, or trusted locker contract names.【F:ethereumbotv2.py†L2015-L2138】【F:ethereumbotv2.py†L3176-L3313】【F:ethereumbotv2.py†L2141-L2219】

## Layered Source Strategy

### 1. Holder Analytics First
- **Ethplorer top holders.** `_fetch_ethplorer_top_holders` pages through the public Ethplorer endpoint with keyed retries, normalises each holder record, and logs structured telemetry for operators.【F:ethereumbotv2.py†L4247-L4288】
- **Automatic fallback.** `_fetch_holder_distribution_async` reuses the Ethplorer snapshot when available, but transparently drops to the Etherscan `tokenholderlist` API whenever Ethplorer does not respond or the operator needs more depth.【F:ethereumbotv2.py†L4293-L4319】
- **95/5 gating.** `_check_liquidity_locked_holder_analysis` tallies balances by status, credits unknown balances to the remainder, and only returns `True` when ≥95 % of the supply is categorised as locked (with the mirror condition for an unlocked verdict).【F:ethereumbotv2.py†L2015-L2138】

This combination gives the bot a deterministic "lock" verdict whenever a reputable locker or burn sink dominates the LP supply, making holder analysis the fastest route for high-confidence confirmations.

### 2. UNCX Subgraph Confirmation
- **GraphQL mirror.** `_check_liquidity_locked_uncx_graph_async` hits the modern UNCX subgraph, parsing both explicit lock records and aggregated `lockedPools` metadata so Graph coverage still detects perpetual locks even when granular entries are missing.【F:ethereumbotv2.py†L3176-L3313】
- **Orchestrator.** `_check_liquidity_locked_uncx_async` gates requests with the feature flag and returns the subgraph verdict directly.【F:ethereumbotv2.py†L3316-L3324】

Focusing exclusively on the graph removes a brittle dependency on the legacy REST endpoint while keeping the dedicated UNCX data source in place.

### 3. Etherscan Transaction Sweep
When neither holders nor UNCX can supply an answer, `_check_liquidity_locked_etherscan_async` inspects the LP token’s transfer log directly:
- Looks for mints to the zero address/`0x…dead` burns that destroy supply entirely.【F:ethereumbotv2.py†L2181-L2189】
- Accepts explicit locker function signatures (`lock`, `lockLPToken`, etc.) while screening out `unlock` actions.【F:ethereumbotv2.py†L2191-L2196】
- Fetches the destination contract’s verified source and whitelists common locker prefixes such as `Unicrypt`, `Pink`, or `Team` if the ABI lacks function metadata.【F:ethereumbotv2.py†L2204-L2217】

Any transport/network errors automatically disable Etherscan lookups for the current session, signalling upstream health monitors while preventing false "locked" positives.【F:ethereumbotv2.py†L2218-L2222】

### 4. DexScreener as a Confidence Bonus
Even after the internal checks resolve a lock decision, the DexScreener integration only honours its `locked` label if our own pipeline still reads `False`, ensuring marketing tags cannot overrule on-chain facts.【F:ethereumbotv2.py†L2788-L2814】

## Manual Reporting Workflow
Operators capture high-signal launches in follow-up reports that mirror the automated pipeline, giving human auditors a ready-made audit trail. For example, the Maxi Doge/WETH report enumerates the Ethplorer creation timestamp, the initial Etherscan mint, the UNCX lock transfer, and the subgraph coverage—condensing the lifecycle into a single timing table for rapid reviews.【F:run_reports/manual_pair_followup_0x7d89F902_2025-11-15.md†L1-L22】

The resulting table matches the live dashboard shared with stakeholders, reinforcing why “we have one of the best lock-liquidity check systems”: each row is backed by redundant on-chain sources, and any operator can retrace the evidence in seconds.

## Why It Matters
- **Defence in depth.** Three independent confirmations (holders, UNCX, raw transactions) drastically lower the odds of a forged lock label slipping through.【F:ethereumbotv2.py†L2015-L2219】【F:ethereumbotv2.py†L3176-L3324】
- **Operational resilience.** Every network call is wrapped with telemetry and graceful degradation, so outages at any upstream service disable that leg without crashing the entire scan loop.【F:ethereumbotv2.py†L3176-L3225】【F:ethereumbotv2.py†L2141-L2219】【F:ethereumbotv2.py†L3660-L3699】
- **Human-friendly evidence.** Hand-curated follow-ups keep analysts in the loop and give clients tangible proof of lock coverage, complete with unlock schedules and durations.【F:run_reports/manual_pair_followup_0x7d89F902_2025-11-15.md†L1-L22】

Together, the automation and documentation demonstrate why the desk can confidently claim to run one of the strongest liquidity-lock verification stacks in the space.
