# Bot log check (2026-01-19)

## Commands used
- `rg -n "0x9b46Fb6076C71D26A299114d05f4a7E2190ce673" /tmp/bot.log.2026-01-19`
- `rg -n "0x27Fb424e345F28A9F2500a8425fA6Df4e0F5D064" /tmp/bot.log.2026-01-19`
- `rg -n "\\[ERROR\\]|Traceback" /tmp/bot.log.2026-01-19 | head -n 40`

## Findings
- The log repeatedly reports `maintenance_loop_error` with `name '_safe_float' is not defined`, indicating a missing helper used during lifecycle snapshot collection.
- The pair `0x9b46Fb6076C71D26A299114d05f4a7E2190ce673` is observed: it was requeued due to DexScreener `not_listed`, then rechecked as high-priority, marked HIGH_RISK, and removed after failing for 10 minutes.
- The pair `0x27Fb424e345F28A9F2500a8425fA6Df4e0F5D064` did not appear in this log snapshot (no match found), suggesting it was not detected during the logged window.
