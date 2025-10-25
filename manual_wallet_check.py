#!/usr/bin/env python3
"""Run safety checks for specific wallets using the SmartWalletTracker.

This utility is meant for manual verification of the wallet risk scoring
pipeline.  It accepts one or more wallet addresses and produces a concise
summary of the tracker output (risk score, buy/sell counts, suspicious
activities, etc.).

Example:

    python manual_wallet_check.py \
        --token 0x123...dead \
        0x95AF4aF910c28E8EcE4512BFE46F1F33687424ce

The script relies on the same environment variables as the main bot:
    * ALCHEMY_URL / INFURA_URL / INFURA_URL_V3 for RPC connectivity
    * ETHERSCAN_API_KEYS or ETHERSCAN_API_KEY for Etherscan access
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Iterable, List, Sequence

from web3 import HTTPProvider, Web3

from wallet_tracker_system import SmartWalletTracker, WalletType, set_notifier


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _load_wallets(addresses: Sequence[str], wallet_file: str | None) -> List[str]:
    wallets: List[str] = []
    if wallet_file:
        with open(wallet_file, "r", encoding="utf-8") as handle:
            for line in handle:
                address = line.strip()
                if address:
                    wallets.append(address)
    wallets.extend(addresses)

    seen = set()
    unique: List[str] = []
    for addr in wallets:
        key = addr.lower()
        if key not in seen:
            seen.add(key)
            unique.append(addr)
    return unique


def _provider_urls(cli_url: str | None) -> List[str]:
    urls = []
    if cli_url:
        urls.append(cli_url)
    urls.extend(
        [
            os.getenv("ALCHEMY_URL"),
            os.getenv("INFURA_URL"),
            os.getenv("INFURA_URL_V3"),
            os.getenv("ALCHEMY_URL_BACKUP"),
            os.getenv("INFURA_URL_BACKUP"),
            os.getenv("INFURA_URL_EMERGENCY_1"),
            os.getenv("INFURA_URL_EMERGENCY_2"),
        ]
    )
    return [url for url in urls if url]


def _build_web3(urls: Iterable[str]) -> Web3:
    last_error: Exception | None = None
    for url in urls:
        try:
            provider = HTTPProvider(url, request_kwargs={"timeout": 20})
            w3 = Web3(provider)
            if w3.is_connected():
                logging.info("Connected to provider %s", url)
                return w3
            last_error = RuntimeError(f"Failed handshake with {url}")
        except Exception as exc:  # pragma: no cover - connection errors
            last_error = exc
            logging.debug("Provider connection error for %s: %s", url, exc)

    raise RuntimeError("Unable to connect to any Ethereum RPC provider") from last_error


def _load_etherscan_keys(cli_keys: Sequence[str] | None) -> List[str]:
    if cli_keys:
        return [key.strip() for key in cli_keys if key.strip()]

    raw = os.getenv("ETHERSCAN_API_KEYS")
    if not raw:
        raw = os.getenv("ETHERSCAN_API_KEY", "")
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    return keys


def _make_key_getter(keys: Sequence[str]):
    if not keys:
        logging.warning(
            "No Etherscan API keys configured; lookups will likely return empty data"
        )
        return lambda: ""

    index = 0

    def _get_next() -> str:
        nonlocal index
        key = keys[index]
        index = (index + 1) % len(keys)
        return key

    return _get_next


def _format_timestamp(ts: int) -> str:
    if not ts:
        return "unknown"
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_activity(wallet: str, activity) -> None:
    print(f"\nWallet: {wallet}")
    print(f"  Type: {activity.wallet_type.value}")
    print(f"  Risk score: {activity.risk_score}/100")
    print(f"  ETH balance: {activity.eth_balance:.6f}")
    print(f"  Token balance: {activity.token_balance:.6f}")
    print(f"  Total buys: {activity.total_buys}")
    print(f"  Total sells: {activity.total_sells}")
    print(f"  Last activity: {_format_timestamp(activity.last_activity)}")

    if activity.suspicious_activities:
        print("  Suspicious activities:")
        for flag in activity.suspicious_activities:
            print(f"    • {flag}")
    else:
        print("  Suspicious activities: none detected")

    if activity.marketing_spends:
        print(
            f"  Marketing spends: {len(activity.marketing_spends)} (showing up to 3 most recent)"
        )
        for spend in activity.marketing_spends[:3]:
            ts = _format_timestamp(spend.get("timestamp", 0))
            dest = spend.get("type", "unknown")
            value = spend.get("value_eth", 0.0)
            print(f"    • {value:.4f} ETH to {dest} on {ts}")
    else:
        print("  Marketing spends: none recorded")


def _activity_to_dict(activity) -> dict:
    return {
        "wallet_type": activity.wallet_type.value,
        "risk_score": activity.risk_score,
        "eth_balance": activity.eth_balance,
        "token_balance": activity.token_balance,
        "total_buys": activity.total_buys,
        "total_sells": activity.total_sells,
        "last_activity": activity.last_activity,
        "marketing_spends": activity.marketing_spends,
        "suspicious_activities": activity.suspicious_activities,
    }


async def _run(token: str, wallets: Sequence[str], wallet_type: WalletType, args) -> None:
    urls = _provider_urls(args.provider)
    if not urls:
        raise RuntimeError(
            "No RPC provider configured. Use --provider or set ALCHEMY_URL/INFURA_URL."
        )

    w3 = _build_web3(urls)
    keys = _load_etherscan_keys(args.etherscan_key)
    tracker = SmartWalletTracker(w3, _make_key_getter(keys))

    set_notifier(sync_fn=lambda message: logging.info("NOTIFY: %s", message))

    activities = {}
    for wallet in wallets:
        activity = await tracker.analyze_wallet_activity(wallet, token, wallet_type)
        activities[wallet] = activity

    if args.output == "json":
        serialised = {wallet: _activity_to_dict(activity) for wallet, activity in activities.items()}
        json.dump(serialised, sys.stdout, indent=2)
        print()
    else:
        print("\n=== Wallet Safety Report ===")
        for wallet, activity in activities.items():
            _print_activity(wallet, activity)
        print("\nAll checks completed. If data returned above, the tracker is functioning.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wallets", nargs="*", help="Wallet addresses to analyse")
    parser.add_argument(
        "--wallet-file",
        help="Optional file containing one wallet address per line",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Token contract address used for wallet activity analysis",
    )
    parser.add_argument(
        "--wallet-type",
        choices=[t.value for t in WalletType],
        default=WalletType.UNKNOWN.value,
        help="Assumed wallet role (affects risk scoring)",
    )
    parser.add_argument(
        "--provider",
        help="Override RPC provider URL (defaults to env vars)",
    )
    parser.add_argument(
        "--etherscan-key",
        action="append",
        help="Override Etherscan API key(s); can be supplied multiple times",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    wallets = _load_wallets(args.wallets, args.wallet_file)
    if not wallets:
        raise SystemExit("At least one wallet address must be provided")

    wallet_type = WalletType(args.wallet_type)

    try:
        asyncio.run(_run(args.token, wallets, wallet_type, args))
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        print("Interrupted")
    except Exception as exc:
        logging.error("Safety check failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
