# Manual Pair Follow-Up: 0xD09e30ED1792f8CC4a9CCdc6dbB4BD254F999B79 (manual request)

## Status
- Manual review requested for this pair. On-chain creation, mint, and lock details are not included here because no authoritative transaction history was captured during this follow-up.
- When definitive data becomes available (Ethplorer, subgraph, or direct RPC traces), update this file with creation timestamps, initial LP mint amounts, and any locker transactions.

## Optional Self-Derived Top-Holder Snapshot (informational only)
- The optional derivation path (`ENABLE_SELF_DERIVED_TOKEN_HOLDERS=1`) can rebuild a holder table from LP transfer history (Etherscan `tokentx` scans or a third-party subgraph). Use it only for dashboards; it does **not** affect the liquidity-lock decision logic.
- No derived holder data is embedded here because the transfer history was not fetched in this manual run. Re-run the derivation pipeline and paste the resulting table below when available.

| Rank | Holder | Balance (raw) | Balance (LP tokens) | Notes |
| --- | --- | --- | --- | --- |
| — | — | — | — | Derivation pending (awaiting transfer history) |

## Next Steps
- Capture the pair’s creation and initial mint transactions.
- Confirm whether liquidity was locked (locker address, amount, and unlock date), or document absence of any lock.
- Re-run the self-derived holder reconstruction and add the resulting snapshot to the table above for visibility.
