# Manual Pair Follow-Up: 0xBC95855c1C1F45cBB8EA0Eb0D8e8017c3D02c0E5 (Vacox / WETH)

## Recap: Launch and Lock Evidence
- Launch mint minted **44,721.359549995793927183 UNI-V2** to the deployer wallet in tx `0x823c…5573` (block **23848767**) at **2025-11-21 17:09:59 UTC**.
- The deployer moved the full LP balance to the Unicrypt locker `0x71B5…7641` in tx `0x7a61…3011` (block **23850638**) at **2025-11-21 23:28:23 UTC**, encoding an unlock date of **2026-02-22 00:00:00 UTC**. The mint and lock amounts matched, so coverage was effectively **100%** of supply.

## Optional Self-Derived Top-Holder Snapshot (informational only)
- With `ENABLE_SELF_DERIVED_TOKEN_HOLDERS=1`, the bot rebuilt holder balances directly from Etherscan `tokentx` history (ascending, first 200 transfers by default). This snapshot is surfaced for dashboards only and is **not** used for liquidity lock decisions.

| Rank | Holder | Balance (raw) | Balance (LP tokens) | Notes |
| --- | --- | --- | --- | --- |
| 1 | `0x71B5759d73262FBb223956913ecF4ecC51057641` | 44,721,359,549,995,793,927,183 | 44,721.359549995793927183 | Unicrypt locker receiving the full mint in `0x7a61…3011` |

## Interpretation
- The derived view confirms a single holder with the entire LP supply—the same Unicrypt locker already identified in the launch timeline—so there is no evidence of unlocked residual balances.
