# Manual Pair Follow-Up: 0x8876ef58c1917C4bfD773b53108de6f8aEfC8869 (Fusaka / WETH)

## Pair Release Timing (Ethplorer)
- Ethplorer records the Uniswap V2 LP token coming online at **2025-08-11 17:00:11 UTC** (`creationTimestamp` 1,754,931,611) via transaction `0xa863…f200`, establishing the pair’s public launch moment.【787700†L1-L71】【9be7ae†L1-L26】

## Liquidity Custody Evidence (Etherscan & UNCX)
- The same launch transaction minted **18.230334610203948846 UNI-V2** to deployer `0xd7c1…d521`, seeding the opening liquidity position.【9be7ae†L13-L26】【2b9bcd†L1-L8】
- Roughly **2 minutes 36 seconds** later, transaction `0x0feb…c62e` sent **17.230334610203948846 UNI-V2** (≈99.99 % of supply) to the burn address `0x0000…dead`, effectively renouncing custody of the LP tokens instead of locking them in UNCX.【f6c9c7†L1-L19】【2b9bcd†L1-L8】
- Current holder data corroborates the burn outcome: Ethplorer lists the dead wallet with ~99.99 % of UNI-V2 while the only live wallet retains ~0.01 %.【af8550†L1-L15】
- Attempts to query Unicrypt’s public locker APIs for this pair return **HTTP 503 Service Unavailable**, and there is no corresponding UNCX lock record to cite—consistent with the LP supply being burned rather than escrowed.【2a996b†L1-L11】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-08-11 17:00:11 UTC | Ethplorer `creationTimestamp`【787700†L1-L71】 |
| Initial LP mint | 18.230334610203948846 UNI-V2 to deployer | Etherscan `tokentx`【9be7ae†L13-L26】 |
| Burn event | 2025-08-11 17:02:47 UTC – 17.230334610203948846 UNI-V2 to `0x…dead` | Etherscan `tokentx`【f6c9c7†L1-L19】 |
| UNCX status | No lock entry; API responds 503 | UNCX REST probe【2a996b†L1-L11】 |
| Holder snapshot | 99.99 % burned, 0.01 % live | Ethplorer holders list【af8550†L1-L15】 |
