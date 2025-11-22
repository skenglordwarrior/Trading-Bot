"""Utility to fetch pair creation time and liquidity lock hints via Etherscan.

This script retrieves the contract creation block for a Uniswap-style pair
contract and runs the bot's on-chain liquidity-lock detection helpers to
summarise whether/when liquidity was locked.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

from etherscan_config import load_etherscan_base_urls, load_etherscan_keys
from ethereumbotv2 import _check_liquidity_locked_etherscan_async


def _pick_etherscan_credentials() -> Tuple[str, Tuple[str, ...]]:
    keys = load_etherscan_keys()
    # Prefer the legacy /api endpoint when present because some niche
    # operations (e.g. getcontractcreation) are not exposed on the v2 path.
    urls = tuple(sorted(load_etherscan_base_urls(), key=lambda u: "v2" in u))
    if not keys:
        raise SystemExit("No Etherscan API keys configured.")
    if not urls:
        raise SystemExit("No Etherscan API base URLs configured.")
    return keys[0], urls


def fetch_creation_timestamp(
    pair_addr: str, api_key: str, base_urls: Tuple[str, ...]
) -> Tuple[Optional[int], Optional[int]]:
    """Return (block_number, timestamp) for the pair contract creation."""

    for base_url in base_urls:
        creation_params = {
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": pair_addr,
            "apikey": api_key,
        }
        if "v2" in base_url:
            creation_params["chainid"] = 1

        resp = requests.get(base_url, params=creation_params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        result = payload.get("result") or []
        first_entry = result[0] if result else {}
        entry = first_entry if isinstance(first_entry, dict) else {}

        block_number: Optional[int] = None
        timestamp: Optional[int] = None
        try:
            block_raw = entry.get("blockNumber")
            block_number = int(block_raw) if block_raw else None
        except (TypeError, ValueError):
            block_number = None

        try:
            ts_raw = entry.get("timeStamp")
            timestamp = int(ts_raw) if ts_raw else None
        except (TypeError, ValueError):
            timestamp = None

        if block_number and timestamp is None:
            block_params = {
                "module": "block",
                "action": "getblockreward",
                "blockno": block_number,
                "apikey": api_key,
            }
            if "v2" in base_url:
                block_params["chainid"] = 1

            block_resp = requests.get(base_url, params=block_params, timeout=15)
            block_resp.raise_for_status()
            block_payload = block_resp.json()
            try:
                ts_val = (block_payload.get("result") or {}).get("timeStamp")
                timestamp = int(ts_val) if ts_val is not None else None
            except (TypeError, ValueError):
                timestamp = None

        if block_number or timestamp:
            return block_number, timestamp

    return None, None


def _format_timestamp(ts: Optional[int]) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect pair creation and liquidity lock status")
    parser.add_argument("pair_address", help="Pair (LP token) contract address")
    args = parser.parse_args()

    api_key, base_urls = _pick_etherscan_credentials()

    block_number, creation_ts = fetch_creation_timestamp(args.pair_address, api_key, base_urls)
    if block_number and creation_ts:
        print(f"Pair created in block {block_number} at {_format_timestamp(creation_ts)}")
    else:
        print("Could not determine creation block/time from Etherscan")

    locked, details = asyncio.run(_check_liquidity_locked_etherscan_async(args.pair_address))
    if locked:
        locked_at = getattr(details, "locked_at", None) if details else None
        unlock_at = getattr(details, "unlock_at", None) if details else None
        coverage = getattr(details, "coverage_pct", None) if details else None
        parts = ["Liquidity appears locked"]
        if coverage is not None:
            parts.append(f"coverage={coverage:.2%}")
        if locked_at:
            parts.append(f"locked_at={_format_timestamp(locked_at)}")
        if unlock_at:
            parts.append(f"unlock_at={_format_timestamp(unlock_at)}")
        print("; ".join(parts))
    else:
        print("No definitive liquidity lock detected")


if __name__ == "__main__":
    main()
