# Manual Pair Report: 0x9796B9E60587495D27eB251a832C1D397FC2e028 (Maxi Doge / WETH)

## Latest Bot Scan (2025-11-14)
- Running `check_pair_criteria` against the Maxi Doge (`0x9796…2e028`) and WETH (`0xC02a…56Cc2`) pair (`0x7d89…C0Af1`) returns **13 / 14** checks passed. The lone miss comes from the `no_private_sale` rule, while liquidity, volume, honeypot tests, renounce status, and contract verification all pass.【4251df†L1-L16】【4251df†L18-L31】
- DexScreener supplied healthy 24 h activity (1,768 trades, $6.6 M volume) and ~$823 k of liquidity, with the bot confirming renounce status and liquidity locks on-chain.【4251df†L16-L31】

## Pair Launch Timing
- Ethplorer reports the Uniswap V2 LP token was created at **2025-11-14 00:47:47 UTC** (`creationTimestamp` 1,763,081,267) by the Maxi Doge deployer, establishing the pool’s release time.【792ea7†L1-L15】

## Liquidity Lock Details (Ethplorer, Etherscan, UNCX)
- The initial LP mint at transaction `0x2859…3796` minted **0.7 UNI-V2** to the deployer wallet immediately after pool creation.【5f706e†L1-L29】
- A follow-up UniswapV2Locker interaction (`0x7b9b…afea`) sent **0.699999999999999 UNI-V2** to the locker address `0x663a…b214` at **2025-11-14 02:38:47 UTC**, with the standard **0.006999999999999 UNI-V2** fee forwarded to the withdrawer wallet.【5f706e†L29-L60】
- The UNCX subgraph lists a single active lock covering **0.69299999999999901 UNI-V2** (≈99 % of supply), pairing **4,046,682.5718 MAXI** with **127.3411 WETH** and scheduling unlock for **2025-11-30 02:18:00 UTC** (`unlockDate` 1,764,469,080).【7cadc7†L1-L4】
- Comparing timestamps shows liquidity was locked 1 h 51 m after launch and will remain locked for **15 days, 23 hours, 39 minutes, 13 seconds** (difference between 1,763,087,927 and 1,764,469,080).【5f706e†L29-L45】【7cadc7†L1-L4】【424b24†L1-L6】

## Current On-Chain Liquidity Snapshot
- Calling `getReserves` at block timestamp 1,763,130,599 (2025-11-14 14:29:59 UTC) returned **4,087,558.1533 MAXI** versus **526.8578 WETH**, confirming substantial liquidity remains in the pool today.【7f7d94†L1-L1】【5e8cf1†L1-L5】【707a3b†L1-L3】【fd25ba†L1-L6】
- Maxi Doge carries 9 decimals according to Ethplorer, so the base reserve equates to roughly 4.09 million tokens held inside the LP.【6a4837†L1-L20】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-14 00:47:47 UTC | Ethplorer `creationTimestamp`【792ea7†L1-L15】 |
| Liquidity lock executed | 2025-11-14 02:38:47 UTC | Etherscan `tokentx` entry【5f706e†L29-L45】 |
| Lock amount | 0.699999999999999 UNI-V2 | Transfer to UniswapV2Locker【5f706e†L29-L45】 |
| Lock coverage | 98.99999999999986 % (0.69299999999999901 UNI-V2) | UNCX subgraph `lockedLiquidity`/`lockedPercent`【7cadc7†L1-L4】 |
| Unlock scheduled | 2025-11-30 02:18:00 UTC | UNCX subgraph `unlockDate`【7cadc7†L1-L4】 |
| Lock duration | 15d 23h 39m 13s | Derived from timestamps【5f706e†L29-L45】【7cadc7†L1-L4】【424b24†L1-L6】 |
| Current bot passes | 13 / 14 (private sale flag) | `check_pair_criteria` output【4251df†L1-L31】 |
| Current reserves | ~4.09 M MAXI vs 526.86 WETH | `getReserves` output & token decimals【7f7d94†L1-L1】【5e8cf1†L1-L5】【fd25ba†L1-L6】【6a4837†L1-L20】 |

