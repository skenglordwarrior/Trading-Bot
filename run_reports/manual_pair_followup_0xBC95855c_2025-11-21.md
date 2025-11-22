# Manual Pair Follow-Up: 0xBC95855c1C1F45cBB8EA0Eb0D8e8017c3D02c0E5 (Vacox / WETH)

## Pair Release Timing (Ethplorer)
- Ethplorer shows the Uniswap V2 LP token created at **2025-11-21 17:09:59 UTC** (`creationTimestamp` 1,763,744,999) by the Vacox deployer (`creatorAddress` `0xd81f…58af`) in transaction `0x823c…5573`.【d79967†L1-L15】【3f903b†L1-L3】

## Liquidity Lock Evidence (Etherscan & Locker Logs)
- The launch transaction (`0x823c…5573`, block **23848767**) minted **44,721.359549995793927183 UNI-V2** to the deployer wallet `0x9a2a…cfe5`, establishing the initial Vacox/WETH liquidity position.【6c2364†L4-L35】
- A follow-up locker call (`0x7a61…3011`, block **23850638**) transferred that full **44,721.359549995793927183 UNI-V2** stake from the deployer to the Unicrypt locker `0x71b5…7641` at **2025-11-21 23:28:23 UTC**.【6c2364†L36-L57】【a3f939†L1-L3】
- The locker event encodes an `unlockDate` of `0x699a4700` → **2026-02-22 00:00:00 UTC**, leaving the entire LP supply escrowed for roughly **93 days** after launch.【a2d4e3†L18-L29】【3ced1c†L1-L6】【a1b2b2†L1-L3】
- Because the mint and lock amounts match the total supply, **100% of the LP tokens were locked** in that single transaction.【6c2364†L20-L57】【3ced1c†L1-L3】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-21 17:09:59 UTC | Ethplorer `creationTimestamp`【d79967†L1-L15】【3f903b†L1-L3】 |
| Initial LP mint | 44,721.359549995793927183 UNI-V2 @ 2025-11-21 17:09:59 UTC | Etherscan `tokentx` entry【6c2364†L4-L35】 |
| Liquidity lock execution | 2025-11-21 23:28:23 UTC | Locker transfer + tx timestamp【6c2364†L36-L57】【a3f939†L1-L3】 |
| Locked amount / coverage | 44,721.359549995793927183 UNI-V2 locked (≈100% of supply) | Supply vs lock delta【6c2364†L20-L57】【3ced1c†L1-L3】 |
| Unlock schedule | 2026-02-22 00:00:00 UTC | Locker log `unlockDate`【a2d4e3†L18-L29】【a1b2b2†L1-L3】 |

