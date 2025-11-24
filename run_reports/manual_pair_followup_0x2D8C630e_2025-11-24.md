# Manual Pair Runtime Report: 0x2D8C630e9F9207FA1c089665da95A5f14Dfd3D33

## Pair Metadata
- **Dex / Pair:** Uniswap v2 pool at `0x2D8C630e9F9207FA1c089665da95A5f14Dfd3D33` for Bitcoin 6900 (`0x5cAa5C27…3b1`) vs WETH (`0xC02a…56Cc2`).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L2-L45】
- **DexScreener snapshot:** price ≈ $0.0500, liquidity ≈ $541k, 24h volume ≈ $3.28M, with 820 buys / 191 sells in the last 24h (1,011 trades total).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L16-L46】
- **Pool lifecycle:** Pair created at 2025-11-23 19:24:35 UTC.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L31-L42】

## Passing Score (Bot Criteria)
- `check_pair_criteria` returned **12 / 14** passes for this pair.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L5-L114】
- **Failed gates:**
  - `risk_score` (contract marked `HIGH_RISK`, `riskScore = 10`, exceeds <10 threshold).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L48-L112】
  - `no_private_sale` (private sale heuristics detect presale/large transfers tied to token and pair addresses).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L83-L114】
- **Risk flags surfaced:** owner functions enabled; trading can be paused; auto-liquidity add present; owner activity seen; Slither unavailable in this run (not installed).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L50-L75】 Owner is zero address and contract is renounced, but the bot still classifies the contract as high risk because of these flags and the presale detection.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L77-L114】

## Liquidity & Lock Status
- **Lock verdict:** Liquidity is marked **locked** via the UNCX graph with ~99.0% coverage (0.7000 LP tokens) locked from 2025-11-23 20:17:47 UTC until 2025-12-09 20:16:00 UTC (≈16.0 days).【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L31-L39】
- **Safety signals:** No recent liquidity removals detected; honeypot checks pass for both sides.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L96-L113】
- **On-chain concentration:** Holder concentration reported at ~2.51%; no smart-money wallets flagged.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L90-L94】

## Takeaways
- The pool meets all volume/liquidity/honeypot gates and shows a near-full UNCX lock through early December 2025, but the bot still rejects auto-trading because of the **high risk score** and **presale detection**. Monitor how the risk score evolves (e.g., if presale dynamics settle) before considering this pair for passing status.【F:run_reports/manual_pair_followup_0x2D8C630e_data.json†L5-L114】
