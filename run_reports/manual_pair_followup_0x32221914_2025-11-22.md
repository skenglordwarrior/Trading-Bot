# Manual Pair Follow-Up: 0x3222191466156fdc1E48658d2b5EFC37f5682861 (manual request)

## Status
- 0x3222…2861 is the **base token** (ETHDOGS), not the LP pair address. Dexscreener search shows its active Uniswap V2 pair is 0xD09e30ED1792f8CC4a9CCdc6dbB4BD254F999B79 created on 2025-11-22 16:47:11 UTC.
- UNCX graph reports one active lock on that pair: 98.99999999999986% of the LP (0.700035713374681059 LP tokens ≈ $165.6k, holding 8.71M ETHDOGS and 59.63 WETH) locked at 2025-11-22 16:56:11 UTC, unlocking at 2025-12-18 12:43:00 UTC.

## UNCX lock evidence (pair 0xD09e30ED1792f8CC4a9CCdc6dbB4BD254F999B79)
- Lock ID: 0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214… on the ETHDOGS/WETH pair.
- Locked liquidity: 0.700035713374681059 LP (98.99999999999986% coverage), holding 8,707,265.6464 ETHDOGS and 59.6324 WETH (~$165.6k core USD at lock time).
- Lock window: locked at 2025-11-22 16:56:11 UTC; unlock scheduled for 2025-12-18 12:43:00 UTC.

## Optional Self-Derived Top-Holder Snapshot (informational only)
- The optional derivation path (`ENABLE_SELF_DERIVED_TOKEN_HOLDERS=1`) can rebuild a top-holder table from LP transfer history (Etherscan `tokentx` scans or a third-party subgraph/indexer). Use it for dashboards only; it does **not** influence the liquidity-lock decision tree.
- For visibility, the current lock entry shows the Unicrypt locker holding essentially all LP tokens; rerun the derivation if you want full holder coverage from transfer history.

| Rank | Holder | Balance (raw) | Balance (LP tokens) | Notes |
| --- | --- | --- | --- | --- |
| 1 | 0x663a5c229c09b049e36dcc11a9b0d4a8eb9db214 | 0.700035713374681059 | 0.700035713374681059 | UNCX lock for ETHDOGS/WETH (≈99% of LP supply) |

## Next Steps
- Keep this file linked to the correct LP pair (0xD09e…B79) when adding future observations.
- Capture any additional liquidity-add events and monitor the unlock on 2025-12-18.
- Refresh the self-derived holder reconstruction after notable transfers to show any LP movement post-unlock.
