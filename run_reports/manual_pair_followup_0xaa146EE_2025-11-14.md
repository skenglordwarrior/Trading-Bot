# Manual Pair Follow-Up: 0xaa146EE424cdeccF76Dc198d57fF9b22C6477F8d (Little Pepe / WETH)

## Latest Bot Scan (2025-11-14)
- Running `check_pair_criteria` against the pair with Little Pepe (`0x6462…AC41`) and WETH (`0xC02a…56Cc2`) returns **0 / 14** passes. The helper reports `dexscreener_reason = "not_listed"`, marks the failure as transient, and would requeue the pair once metadata becomes available again.【7690b2†L1-L12】
- DexScreener’s pair endpoint now responds with `pairs: null`, confirming the listing has been removed and explaining the metadata miss in the automated check.【ce0e58†L1-L6】

## Pair Launch Timing
- Ethplorer’s `getTokenInfo` call for the LP token shows it was created at **2025-10-25 06:33:35 UTC** (`creationTimestamp` 1,761,374,015) via transaction `0xb2ba…6dcfed`. Total supply remains 0.7 UNI-V2 with only two holders recorded, matching the original lock footprint.【d3fdab†L1-L20】

## Liquidity Lock Details (Non-DexScreener Sources)
- Etherscan V2 `tokentx` history captures the LP mint to the deployer (`0x8fC4…c162`) and the subsequent lock transaction that sent **0.699999999999999 UNI-V2** to `UniswapV2Locker` (`0x663a…b214`) on **2025-10-25 06:44:59 UTC** (`timeStamp` 1,761,374,699).【277d6f†L1-L21】【bab21e†L1-L22】
- The same transaction forwarded **0.0069999999999999 UNI-V2** to the withdrawer wallet `0xd45d…8dc1c`, representing the locker fee.【80b95d†L1-L22】
- The Unicrypt (UNCX) Graph entry for this pool lists a single lock with `unlockDate = 1762843380`, i.e. **2025-11-11 06:43:00 UTC**, matching the schedule encoded on-chain even though the current snapshot reports zero active liquidity (the LP tokens remain custodied by the locker until unlock).【25b655†L1-L24】
- Translating the timestamps shows the LP token was minted at 06:33:35 UTC, locked at 06:44:59 UTC, and scheduled to unlock at 06:43:00 UTC on 2025-11-11—a duration of **16 days, 23 hours, 58 minutes, 1 second**.【d3fdab†L1-L20】【bab21e†L1-L22】【2d4c22†L1-L15】
- The bot’s direct `_check_liquidity_locked_uncx_async` helper currently falls back to **False** because the public UNCX REST endpoint is returning HTTP 503, so automated runs rely on the Graph or other sources for confirmation.【0e77b5†L1-L2】

## Current Liquidity Snapshot
- On-chain calls against the pair show reserves of `0x04ebdc5d9ec88dcf` Little Pepe and `0x1aa75090f532f3ea` WETH at block timestamp `0x6908c51f`. `token0()` returns Little Pepe (`0x6462…AC41`) with 9 decimals, while WETH exposes the standard 18 decimals.【2e6460†L1-L4】【8e1672†L1-L3】【d208f3†L1-L3】【f31875†L1-L3】
- Converting those reserves yields approximately **354.6 million LILPEPE** versus **1.92 WETH** in the pool today, underscoring the liquidity collapse compared with launch week.【ed3ece†L1-L5】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-10-25 06:33:35 UTC | Ethplorer `creationTimestamp`【d3fdab†L1-L20】 |
| Liquidity lock executed | 2025-10-25 06:44:59 UTC | Etherscan `tokentx` entry【bab21e†L1-L22】 |
| Lock amount | 0.699999999999999 UNI-V2 | `tokentx` amount to UniswapV2Locker【bab21e†L1-L22】 |
| Locker fee (forwarded) | 0.0069999999999999 UNI-V2 | `tokentx` amount to withdrawer【80b95d†L1-L22】 |
| Unlock scheduled | 2025-11-11 06:43:00 UTC | UNCX Graph `unlockDate`【25b655†L1-L24】 |
| Lock duration | 16d 23h 58m 01s | Derived from timestamps【d3fdab†L1-L20】【bab21e†L1-L22】【2d4c22†L1-L15】 |
| Current bot passes | 0 / 14 (Dex missing) | `check_pair_criteria` output【7690b2†L1-L12】 |
| Current liquidity | ~354.6M LILPEPE vs 1.92 WETH | On-chain reserves & decimals【2e6460†L1-L4】【d208f3†L1-L3】【f31875†L1-L3】【ed3ece†L1-L5】 |

