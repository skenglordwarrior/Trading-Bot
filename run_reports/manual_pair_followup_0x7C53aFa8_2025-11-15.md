# Manual Pair Follow-Up: 0x7C53aFa86978da9E012c016d67CE296767c7a56d (Garfield Rizzmas / WETH)

## Pair Release Timing (Ethplorer)
- Ethplorer identifies the Uniswap V2 LP token as launching at **2025-11-15 10:56:23 UTC** (`creationTimestamp` 1,763,204,183), minted by the Garfield Rizzmas deployer wallet—this marks the public release of the pool.【4fbbb2†L1-L20】【13ec2c†L1-L3】【7e855c†L1-L6】

## Liquidity Lock Evidence (Etherscan & UNCX)
- The initial LP mint transaction (`0xe6f3…f518`) produced **0.707106781186546524 UNI-V2** for the deployer at launch, establishing the starting liquidity position.【3da987†L25-L47】
- Ten minutes later, transaction `0x1242…6fdf` invoked `lockLPToken` on the UniswapV2Locker (UNCX) contract, transferring **0.707106781186546524 UNI-V2** to the locker while routing the standard **0.007071067811865465 UNI-V2** fee to the withdrawer wallet.【3da987†L48-L93】【389e45†L1-L22】【4a6fd8†L1-L27】【7bff0d†L1-L12】
- The encoded `_unlock_date` in the same call resolves to **2025-12-15 11:05:00 UTC**, establishing a **29 day 23 hour 58 minute** lock executed **10 minutes 36 seconds** after launch.【389e45†L1-L22】【7bff0d†L1-L12】【ebbd8c†L1-L13】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-15 10:56:23 UTC | Ethplorer `creationTimestamp`【4fbbb2†L1-L20】 |
| Initial LP mint | 0.707106781186546524 UNI-V2 @ 2025-11-15 10:56:23 UTC | Etherscan `tokentx` entry【3da987†L25-L47】 |
| Liquidity lock execution | 2025-11-15 11:06:59 UTC | Etherscan `tokentx` + `lockLPToken` call【3da987†L48-L93】【389e45†L1-L22】 |
| Locked amount / fee | 0.707106781186546524 UNI-V2 locked / 0.007071067811865465 UNI-V2 fee | Etherscan receipt decoding【4a6fd8†L1-L27】【7bff0d†L1-L12】 |
| Unlock schedule | 2025-12-15 11:05:00 UTC | UNCX locker call `_unlock_date`【389e45†L1-L22】【7bff0d†L1-L12】 |
| Lock duration | 29d 23h 58m | Timestamp difference【ebbd8c†L1-L13】 |
