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
