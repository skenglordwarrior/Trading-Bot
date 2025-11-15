# Manual Pair Follow-Up: 0x7d89F90265244b83018d2FD64B49EDdBE05C0Af1 (Maxi Doge / WETH)

## Pair Release Timing (Ethplorer)
- Ethplorer’s `getTokenInfo` endpoint lists the Uniswap V2 LP token as being created at **2025-11-14 00:47:47 UTC** (`creationTimestamp` 1,763,081,267) by the Maxi Doge deployer wallet, marking the pool’s public release time.【9d26fd†L1-L20】【3c4ad8†L7-L9】

## Liquidity Lock Evidence (Etherscan & UNCX)
- The inaugural LP mint (`hash` `0x2859…3796`) produced **0.699999999999999 UNI-V2** for the deployer address at the same 00:47:47 UTC launch timestamp, confirming initial liquidity provisioning.【da6a35†L4-L38】【3c4ad8†L7-L9】
- Transaction `0x7b9b…afea` moved **0.699999999999999 UNI-V2** from the deployer to `UniswapV2Locker` (`0x663a…b214`) at **2025-11-14 02:38:47 UTC**, while forwarding the standard **0.006999999999999 UNI-V2** fee to the withdrawer wallet (`0xd45d…8dc1c`).【da6a35†L39-L61】【3c4ad8†L8-L11】
- The Unicrypt (UNCX) subgraph shows a single active lock covering **0.69299999999999901 UNI-V2** (~99 % of supply), securing **4,114,065.6783 MAXI** against **125.3521 WETH** and scheduling unlock for **2025-11-30 02:18:00 UTC** (`unlockDate` 1,764,469,080).【b40632†L1-L28】【3c4ad8†L8-L11】
- Comparing the creation, lock, and unlock timestamps yields a **15 day, 23 hour, 39 minute, 13 second** lock duration, initiated 1 hour 51 minutes after launch and set to expire on 2025-11-30 02:18:00 UTC.【3c4ad8†L7-L9】【3c4ad8†L8-L11】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-14 00:47:47 UTC | Ethplorer `creationTimestamp`【9d26fd†L1-L20】 |
| Initial LP mint | 0.699999999999999 UNI-V2 @ 2025-11-14 00:47:47 UTC | Etherscan `tokentx` entry【da6a35†L4-L38】 |
| Liquidity lock execution | 2025-11-14 02:38:47 UTC | Etherscan `tokentx` transfer to UniswapV2Locker【da6a35†L39-L56】【3c4ad8†L8-L11】 |
| Lock coverage | 0.69299999999999901 UNI-V2 (98.99999999999986 %) | UNCX subgraph `lockedLiquidity`/`lockedPercent`【b40632†L1-L28】 |
| Unlock schedule | 2025-11-30 02:18:00 UTC | UNCX subgraph `unlockDate`【b40632†L1-L28】【3c4ad8†L8-L11】 |
| Lock duration | 15d 23h 39m 13s | Timestamp difference【3c4ad8†L7-L9】【3c4ad8†L8-L11】 |

