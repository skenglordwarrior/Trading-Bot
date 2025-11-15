#!/usr/bin/env python3
"""Probe the Etherscan ``tokentx`` endpoint for a specific token or LP pair.

This diagnostic utility exists to quickly reproduce the timeout the bot saw
when fetching token transfer history.  It performs a single ``tokentx``
request using the same API key rotation helpers the production code relies on
and prints a concise summary of the response so engineers can determine
whether Etherscan is reachable and behaving as expected.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any, Iterable

import requests

from etherscan_config import load_etherscan_base_urls, load_etherscan_keys

DEFAULT_TIMEOUT = 20
DEFAULT_OFFSET = 25
DEFAULT_PAGE = 1


class EtherscanError(RuntimeError):
    """Raised when every configured Etherscan endpoint fails."""


def _iso_timestamp(raw: str | int | None) -> str:
    if not raw:
        return "unknown"
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        return "unknown"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _attempt_request(url: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected JSON payload from Etherscan")
    return data


def _first(entries: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in entries:
        return entry
    return None


def probe_tokentx(args: argparse.Namespace) -> dict[str, Any]:
    keys = load_etherscan_keys(args.api_key)
    urls = load_etherscan_base_urls(args.api_url)

    if not keys:
        raise RuntimeError(
            "No Etherscan API keys configured. Set ETHERSCAN_API_KEY(S) or use --api-key."
        )

    if not urls:
        raise RuntimeError(
            "No Etherscan API base URLs configured. Set ETHERSCAN_API_URLS or use --api-url."
        )

    params: dict[str, Any] = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": args.contract_address,
        "page": args.page,
        "offset": args.offset,
        "sort": args.sort,
        "apikey": keys[0],
    }

    if args.address:
        params["address"] = args.address
    if args.start_block is not None:
        params["startblock"] = args.start_block
    if args.end_block is not None:
        params["endblock"] = args.end_block
    if args.chain_id:
        params["chainid"] = args.chain_id

    errors = []
    for url in urls:
        try:
            data = _attempt_request(url, params, args.timeout)
            data["_metadata"] = {"url": url, "key": params.get("apikey")}
            return data
        except Exception as exc:  # pragma: no cover - network errors
            errors.append((url, exc))

    details = "; ".join(f"{url}: {exc}" for url, exc in errors)
    raise EtherscanError(details)


def _render_summary(payload: dict[str, Any]) -> None:
    metadata = payload.get("_metadata", {})
    url = metadata.get("url", "unknown")
    status = payload.get("status", "0")
    message = payload.get("message", "")
    result = payload.get("result", [])
    total = len(result) if isinstance(result, list) else 0

    print(f"Etherscan endpoint: {url}")
    print(f"Status: {status} ({message})")
    print(f"Result entries: {total}")

    if not isinstance(result, list):
        print(f"Raw result payload: {result}")
        return

    first_entry = _first(result)
    if not first_entry:
        print("No transfer entries returned.")
        return

    tx_hash = first_entry.get("hash", "unknown")
    from_addr = first_entry.get("from", "unknown")
    to_addr = first_entry.get("to", "unknown")
    value = first_entry.get("value", "0")
    timestamp = _iso_timestamp(first_entry.get("timeStamp"))

    print("\nFirst transfer:")
    print(f"  hash: {tx_hash}")
    print(f"  from: {from_addr}")
    print(f"  to:   {to_addr}")
    print(f"  value: {value}")
    print(f"  timestamp: {timestamp}")


def _render_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-address",
        required=True,
        help="Token or LP contract address to inspect",
    )
    parser.add_argument(
        "--address",
        help="Optional wallet address to filter transfers",
    )
    parser.add_argument(
        "--start-block",
        type=int,
        help="Optional start block for the query",
    )
    parser.add_argument(
        "--end-block",
        type=int,
        help="Optional end block for the query",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=DEFAULT_PAGE,
        help="Results page to request (default: %(default)s)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_OFFSET,
        help="Number of records per page (default: %(default)s)",
    )
    parser.add_argument(
        "--sort",
        choices=("asc", "desc"),
        default="asc",
        help="Sort order for the results (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--chain-id",
        dest="chain_id",
        default="1",
        help="Etherscan chain identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        help="Override Etherscan API key (can be specified multiple times)",
    )
    parser.add_argument(
        "--api-url",
        action="append",
        help="Override Etherscan base URL (can be specified multiple times)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Dump the full JSON payload instead of a human summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        payload = probe_tokentx(args)
    except Exception as exc:
        print(f"Etherscan request failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _render_json(payload)
    else:
        _render_summary(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
