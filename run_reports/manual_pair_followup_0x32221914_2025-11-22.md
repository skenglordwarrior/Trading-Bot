# Manual Pair Follow-Up: 0x3222191466156fdc1E48658d2b5EFC37f5682861 (manual request)

## Status
- Manual follow-up requested; no authoritative creation or lock transactions were captured in this snapshot.
- Collect Dex/LP creation, initial mint amounts, and any locker transfers when they become available.

## Optional Self-Derived Top-Holder Snapshot (informational only)
- The optional derivation path (`ENABLE_SELF_DERIVED_TOKEN_HOLDERS=1`) can rebuild a top-holder table from LP transfer history (Etherscan `tokentx` scans or a third-party subgraph/indexer). Use it for dashboards only; it does **not** influence the liquidity-lock decision tree.
- Transfer history was not fetched in this manual pass. Re-run the derivation and paste the output below once available.

| Rank | Holder | Balance (raw) | Balance (LP tokens) | Notes |
| --- | --- | --- | --- | --- |
| — | — | — | — | Derivation pending (awaiting transfer history) |

## Next Steps
- Capture pair creation and liquidity-add transactions (amounts, timestamp, sender).
- Confirm whether liquidity is locked (locker address, amount, unlock date), or document lack of any lock.
- Refresh the self-derived holder reconstruction and add the resulting snapshot to the table above for visibility.
