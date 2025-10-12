# Testing the Ethereum Trading Bot
This environment supports short smoke tests of `ethereumbotv2.py`. For example:
```bash
timeout 300 python ethereumbotv2.py
```
Longer executions (e.g., one-hour runs) are not feasible because interactive sessions in
this workspace automatically terminate after several minutes of inactivity. As a result,
any run exceeding this limit would be interrupted by the environment rather than the bot
itself. To perform extended duration tests, run the bot on an environment you control—such
as a dedicated server or local machine—where you can leave the process active for the
desired amount of time.

## Monitoring Long-Running Sessions Outside This Workspace
If you host the bot on your own machine or server, you can monitor it continuously and
troubleshoot issues while it is running:

1. **Run inside a terminal multiplexer:** start the bot inside `tmux` or `screen` so the
   process keeps running even if your SSH session disconnects. Reattach later to inspect
   the live console output.
2. **Persist logs to disk:** add or enable structured logging in `ethereumbotv2.py` (for
   example via Python's built-in `logging` module) and write logs to a rotating file such
   as `logs/bot.log`. Use `tail -f logs/bot.log` to watch activity in real time.
3. **Stream metrics:** export key performance indicators (orders placed, fills, PnL,
   error counts) to a time-series backend such as Prometheus + Grafana or a SaaS
   monitoring provider. Dashboards make it easy to spot anomalies as they happen.
4. **Set up health alerts:** configure simple scripts or services (cron jobs, systemd
   watchdogs, cloud monitors) that ping the bot's heartbeat endpoint or check log files
   for errors. Send notifications via email, Slack, or SMS when thresholds are breached.
5. **Remote debugging:** if you need to inspect the bot's state interactively, attach a
   debugger such as `pdb` or run the process under `debugpy` and connect from VS Code or
   another IDE while the bot continues to execute.

With this setup, you can leave the bot running unattended yet still observe its behavior,
collect evidence for tuning strategies, and intervene quickly if something goes wrong.
