# Manual Pair Follow-Up: 0xcF8a1Dd4aDe08224812d8b8daf194Fa675815687 (Xera / WETH)

## Bot Criteria Snapshot
- Running `check_pair_criteria` against the pair (Xera vs WETH) currently returns **12 / 14** passes. The failing gates are the `risk_score` check (risk score reported as 11) and the `no_private_sale` check (Dex and contract heuristics see a `hasPresale` flag tied to multiple large transfers), so the bot would keep rechecking until those factors clear.【2b54b3†L1-L5】

## Pair Release Timing (Ethplorer)
- Ethplorer records the Uniswap V2 LP token coming online at **2025-11-14 08:35:35 UTC** (`creationTimestamp` 1,763,109,335), minted via the router to the deployer wallet—this marks the pool’s release to the public.【9869dc†L1-L48】【d20b70†L1-L7】

## Liquidity Lock Evidence (Etherscan & UNCX)
- The launch transaction (`0x37ea…5357`) minted **44,721.359549995796 UNI-V2** to deployer `0x4047…44b5`, establishing the opening liquidity position for the Xera/WETH pair.【6b370e†L7-L34】【f648af†L1-L7】
- Roughly **23 hours 19 minutes 36 seconds** after launch, transaction `0xfc97…d0e9` invoked `lock` on the UniswapV2Locker (UNCX) contract, transferring the full **44,721.359549995796 UNI-V2** stake into escrow.【6b370e†L34-L52】【9daf73†L1-L10】【9fc2fb†L11-L34】
- Decoding the same locker call shows an `_unlock_date` of **2026-02-28 00:00:00 UTC**, confirming a **104 day 16 hour 4 minute 49 second** lock window enforced by UNCX.【9fc2fb†L24-L33】【50d6d7†L1-L5】【9daf73†L5-L10】

## Summary Table

| Item | Value | Source |
| --- | --- | --- |
| Pair creation | 2025-11-14 08:35:35 UTC | Ethplorer `creationTimestamp`【9869dc†L1-L48】【d20b70†L1-L7】 |
| Initial LP mint | 44,721.359549995796 UNI-V2 @ 2025-11-14 08:35:35 UTC | Etherscan `tokentx` entry【6b370e†L7-L34】【f648af†L1-L7】 |
| Liquidity lock execution | 2025-11-15 07:55:11 UTC | Etherscan `tokentx` + UNCX locker call【6b370e†L34-L52】【9fc2fb†L11-L34】【9daf73†L1-L10】 |
| Locked amount | 44,721.359549995796 UNI-V2 held by UniswapV2Locker | Etherscan receipt transfer log【9fc2fb†L11-L23】【f648af†L1-L7】 |
| Unlock schedule | 2026-02-28 00:00:00 UTC | UNCX locker call `_unlock_date`【9fc2fb†L24-L33】【50d6d7†L1-L5】 |
| Lock duration | 104d 16h 4m 49s | Timestamp difference【9daf73†L5-L10】 |
