# Lifecycle snapshot check (2026-01-18)

## Source
Analyzed lifecycle snapshots from the uploaded `lifecycle_snapshots.jsonl`.

## Summary
- Total rows: 54.
- Snapshot offsets observed: {60: 26, 300: 11, 1800: 8, 3600: 5, 14400: 2, 21600: 2}.
- DexScreener missing count: 54 (expected when pairs are not yet listed).
- Rows with DexScreener price fields present: 0.

## Interpretation
- The lifecycle pipeline is writing rows on schedule (multiple offsets are present).
- All rows in this sample show `dexscreener_missing=true`, so price fields are blank, which indicates the pairs were not yet listed on DexScreener at those offsets.
- Once DexScreener lists the pairs, the same row structure will include `price_usd`, `market_cap`, `liquidity_usd`, and trade counts.
