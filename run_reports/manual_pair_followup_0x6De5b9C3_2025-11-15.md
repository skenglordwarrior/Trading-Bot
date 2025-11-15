# Manual Pair Follow-Up: 0x6De5b9C34BbB260C9E19fa254090CdCFd86A4a6D (Saylor Moon / WETH)

## Pair Release Timing (Ethplorer)
- Ethplorer records the Uniswap V2 LP token coming online at **2025-11-15 13:11:59 UTC** (`creationTimestamp` 1,763,212,319), minted by the Saylor Moon deployer wallet—this marks the pool’s public release.【e57737†L1-L21】【52a0ae†L1-L9】【77debe†L1-L7】

## Liquidity Lock Evidence (Etherscan & UNCX)
- The launch transaction (`0x8f07…2436a`) minted **0.699999999999999 UNI-V2** to the deployer, establishing the initial liquidity position for the Saylor Moon/WETH pair.【43b946†L6-L34】【42fdc6†L1-L7】
- Five minutes later, transaction `0xb286…3bef` called `lockLPToken` on the UniswapV2Locker (UNCX) contract, forwarding the full **0.699999999999999 UNI-V2** stake and routing the standard **0.00699999999999999 UNI-V2** fee to the fee wallet while escrowing the remaining **0.69299999999999901 UNI-V2** in the locker.【43b946†L34-L64】【5f8fdd†L9-L46】【42fdc6†L1-L7】
- The same on-chain call timestamps the lock at **2025-11-15 13:16:59 UTC**, exactly **5 minutes** after launch.【43b946†L34-L64】【52a0ae†L1-L9】
- The UNCX subgraph shows the resulting lock covering **98.99999999999986 %** of the LP supply, securing **0.69299999999999901 UNI-V2** against **69.4759724167 WETH / 7,244,551.6198 SAYLOR** and scheduling unlock for **2025-12-15 13:14:00 UTC** (`unlockDate` 1,765,804,440)—a **29 day 23 hour 57 minute** escrow window.【0effd3†L1-L23】【54bdb6†L1-L5】【9873e6†L1-L4】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-15 13:11:59 UTC | Ethplorer `creationTimestamp`【e57737†L1-L21】 |
| Initial LP mint | 0.699999999999999 UNI-V2 @ 2025-11-15 13:11:59 UTC | Etherscan `tokentx` entry【43b946†L6-L34】【42fdc6†L1-L7】 |
| Liquidity lock execution | 2025-11-15 13:16:59 UTC | Etherscan `tokentx` + locker logs【43b946†L34-L64】【5f8fdd†L9-L46】 |
| Locked amount / fee | 0.69299999999999901 UNI-V2 locked / 0.00699999999999999 UNI-V2 fee | Locker transfer logs【5f8fdd†L9-L46】【42fdc6†L1-L7】 |
| Unlock schedule | 2025-12-15 13:14:00 UTC | UNCX subgraph `unlockDate`【0effd3†L1-L23】【54bdb6†L1-L5】 |
| Lock duration | 29d 23h 57m | Timestamp difference【54bdb6†L1-L5】【9873e6†L1-L4】 |

