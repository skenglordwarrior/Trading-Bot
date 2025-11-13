# Liquidity Lock Checks – 2025-11-12

| Pair Address | Locked? | Determination Stage | Notes |
| --- | --- | --- | --- |
| `0x5bcebCEe72F13004F1D00D7Da7BF22b082f93f70` | Yes | Holder snapshot (Ethplorer) | Holder analysis met the ≥95% locked threshold, so the pipeline short-circuited with a locked verdict.【0bbbfe†L1-L3】 |
| `0x8876ef58c1917C4bfD773b53108de6f8aEfC8869` | Yes | Holder snapshot (Ethplorer) | Same Ethplorer-based holder distribution satisfied the lock condition.【049e0c†L1-L3】 |
| `0xDDd23787a6B80A794d952f5fb036D0b31A8E6aff` | Yes | Holder snapshot (Ethplorer) | Holder analysis confirmed ≥95% of LP tokens locked.【2a0243†L1-L2】 |
| `0x332A24318d56f9Cca677a242aFF668314492bF80` | Yes | Holder snapshot (Ethplorer) | Ethplorer holder data marked the pool locked without needing fallbacks.【2a0243†L2-L3】 |
| `0x0846F55387ab118B4E59eee479f1a3e8eA4905EC` | Yes | Etherscan heuristics | Holder and Uncx checks were inconclusive (Uncx REST returned HTTP 503), but the Etherscan tokentx history showed LP tokens sent to the burn address, triggering the burn heuristic.【972efd†L1-L2】【ac3dc7†L1-L34】 |

