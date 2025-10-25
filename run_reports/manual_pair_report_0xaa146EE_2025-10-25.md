# Manual Pair Review: 0xaa146EE424cdeccF76Dc198d57fF9b22C6477F8d

## Pair Metadata
- **Dex**: Uniswap v2
- **Base token**: 0x6462e7842B95CdF81041d8FB4F5084210fFFAC41 ("Little Pepe")
- **Quote token**: WETH (0xC02a…56Cc2)
- **DexScreener snapshot**: price ≈ $0.03825, 24h volume ≈ $2.73M, liquidity ≈ $556k.【eaef34†L1-L31】

## Passing Score (Bot Criteria)
- `check_pair_criteria` returned **6 / 13** checks passed.
- Extra context reported: `riskScore: 9999`, contract marked `verified: False`, `contractRenounced: True`, `liquidityUsd: 556,126.85`, `volume24h: 2,726,236.14`, `marketCap: 38,427,894`, no Slither score (contract fetch failed).【93ee44†L5-L11】

## Etherscan Checks
- `check_liquidity_locked_etherscan` → **False**. Attempts to pull token transfer history from Etherscan returned `status: NOTOK`, so the bot disabled Etherscan lookups and treated liquidity as unlocked.【8a7aba†L1-L4】
- Manual requests against the Etherscan v2 API succeed with bundled keys, but the asynchronous client inside the bot cannot reach `api.etherscan.io` (network unreachable via proxy), leading to disabled lookups during automated checks.【14a95d†L1-L8】【1c2502†L1-L3】

## Advanced Contract Verification
- `advanced_contract_check` falls back to **status = ERROR** with `riskScore = 9999` because source retrieval via the bot’s async client fails after the Etherscan endpoint becomes unreachable. Owner resolves to the zero address with ~14,132 ETH (likely stale data) and the contract is considered renounced by heuristic.【14a95d†L1-L8】【0a61b9†L1-L13】
- Fetching the contract source directly via HTTPS (using the bundled key) returns a verified Solidity file (`Token.sol`, compiler v0.8.24).【42894f†L1-L4】

## Manual Wallet Safety Check
- `manual_wallet_check.py --provider https://eth.llamarpc.com` for the pair address (with token context) reports **risk score 0 / 100**, zero recorded buys/sells, and no suspicious flags. The missing Etherscan API key warning indicates the report relied solely on on-chain balance data.【60b3a0†L1-L17】

## Slither Static Analysis
- Retrieved source saved under `run_reports/token_0x6462e784.sol` for offline analysis.【48f62c†L1-L25】
- Installed `slither-analyzer` locally to run detectors.【a55b78†L1-L8】
- `slither slither_tmp --json slither_report.json --solc-disable-warnings` produced **41 detector findings**, including high-impact issues:
  - `arbitrary-send-eth` via `sendETHToFee` forwarding ETH to `_taxWallet`.
  - Multiple `reentrancy-eth` / `reentrancy-no-eth` warnings on `_transfer`, `openTrading`, and `transferFrom`.
  - `divide-before-multiply` risk in `_taxSwapThreshold`/`_maxTaxSwap` initialisation.
  - Naming/immutability hygiene observations.
  Full console output snippet below for reference.【592b19†L1-L84】【202780†L1-L6】
- JSON report stored at `run_reports/slither_report_0x6462e784.json` (41 issues).【202780†L1-L6】

## Observations & Limitations
- Automated bot checks currently fail to leverage Etherscan due to proxy connectivity problems; manual HTTP requests work, suggesting the async client needs proxy adjustments.
- Liquidity lock status could not be confirmed without reliable Etherscan history.
- Slither highlighted several critical findings (arbitrary ETH sends, reentrancy); these should be reviewed before deeming the project safe.

