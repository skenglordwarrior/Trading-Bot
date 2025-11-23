# Manual Pair Follow-Up: 0xD09e30ED1792f8CC4a9CCdc6dbB4BD254F999B79 (manual request)

## Status
- Active Uniswap V2 pair for ETHDOGS/WETH created on 2025-11-22 16:47:11 UTC.
- UNCX graph shows a single lock covering 98.99999999999986% of LP supply (0.700035713374681059 LP ≈ $165.6k) holding 8.71M ETHDOGS and 59.63 WETH; lock started 2025-11-22 16:56:11 UTC and unlocks 2025-12-18 12:43:00 UTC.

## UNCX lock evidence
- Lock ID: 0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214… on this pair.
- Locked liquidity: 0.700035713374681059 LP (98.99999999999986% coverage), holding 8,707,265.6464 ETHDOGS and 59.6324 WETH (~$165.6k core USD at lock time).
- Lock window: locked at 2025-11-22 16:56:11 UTC; unlock scheduled for 2025-12-18 12:43:00 UTC.

## Optional Self-Derived Top-Holder Snapshot (informational only)
- The optional derivation path (`ENABLE_SELF_DERIVED_TOKEN_HOLDERS=1`) can rebuild a holder table from LP transfer history (Etherscan `tokentx` scans or a third-party subgraph/indexer). Use it for dashboards only; it does **not** affect lock decisions.
- Current lock data shows the Unicrypt locker effectively holds the entire LP supply; rerun the derivation if you want granular post-unlock distribution.

| Rank | Holder | Balance (raw) | Balance (LP tokens) | Notes |
| --- | --- | --- | --- | --- |
| 1 | 0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214 | 0.700035713374681059 | 0.700035713374681059 | UNCX lock (≈99% of LP supply) |

## Next Steps
- Monitor liquidity-add events and the scheduled unlock on 2025-12-18.
- Refresh the self-derived holder snapshot after unlock or any notable LP transfers.
