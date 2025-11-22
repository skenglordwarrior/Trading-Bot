# Liquidity verification pipeline

This note documents how `ethereumbotv2.py` decides whether a Uniswap pair has locked liquidity and clarifies where DexScreener fits into the workflow.

## Locked-liquidity decision tree
1. **LP-holder snapshot.** `_check_liquidity_locked_holder_analysis` builds a snapshot of the LP token supply and classifies each holder using Ethplorer data only. If at least 95% of the supply sits in trusted lockers/burn sinks it returns `True`; if at least 5% is provably unlocked it returns `False`. Any other outcome returns `None` so downstream checks can keep digging. An optional self-derived view can reconstruct top holders from transfer events for dashboards but it is never fed into this decision gate.
2. **UNCX confirmations.** `_check_liquidity_locked_uncx_async` queries the UNCX GraphQL subgraph when the holder snapshot is inconclusive. It returns `True` as soon as it sees an active lock entry and `False` when the feed can prove the opposite; otherwise it bubbles up `None`.
3. **Direct Etherscan sweep.** `_check_liquidity_locked_etherscan_async` invokes the steps above in order and finally inspects recent `tokentx` traces when both are inconclusive. Explicit burns, locker contract names, or direct `lock*` calls flip the verdict to `True`; the helper returns `False` otherwise.

## How DexScreener is used
* `fetch_dexscreener_data` calls `_check_liquidity_locked_etherscan_async` up front and stores the resulting `locked` boolean inside the DexScreener payload. DexScreener's own `"locked"` label is only applied as an advisory add-on when our internal checks could not prove a lock but DexScreener marked it. The downstream pair filters (`check_pair_criteria`) still evaluate liquidity/volume/trade thresholds even if DexScreener is offline, because they immediately requeue pairs that are missing data instead of trusting third-party badges.
* As a result, DexScreener metadata helps with pricing and activity heuristics, but the actual locked-liquidity verdict is governed by the internal holder → UNCX → Etherscan cascade described above.
