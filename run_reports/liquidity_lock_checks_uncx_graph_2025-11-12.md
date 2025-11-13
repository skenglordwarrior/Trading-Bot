# Uncx Subgraph Liquidity Checks – 2025-11-12

| Pair Address | Uncx Graph Verdict | Notes |
| --- | --- | --- |
| `0x5bcebCEe72F13004F1D00D7Da7BF22b082f93f70` | Locked | `lockedLiquidity` entries totaling positive liquidity with future unlock timestamps returned `True` from `_check_liquidity_locked_uncx_graph_async`.【ae1bf0†L2-L3】 |
| `0x8876ef58c1917C4bfD773b53108de6f8aEfC8869` | Inconclusive | Subgraph response contained no `pool` object or `locks` entries for this pair, so the helper returned `None`.【fd2d8a†L1-L1】【730dfb†L1-L1】 |
| `0xDDd23787a6B80A794d952f5fb036D0b31A8E6aff` | Locked | Active `lockedLiquidity` entries produced a positive verdict from the Uncx graph helper.【fd2d8a†L1-L2】 |
| `0x332A24318d56f9Cca677a242aFF668314492bF80` | Inconclusive | Subgraph query returned an empty `locks` array and `pool: null`, yielding `None`.【fd2d8a†L1-L3】【730dfb†L1-L2】 |
| `0x0846F55387ab118B4E59eee479f1a3e8eA4905EC` | Inconclusive | The Uncx subgraph reported no data for the pool, so the check could not confirm a lock.【0cb6b5†L1-L1】【730dfb†L1-L3】 |

