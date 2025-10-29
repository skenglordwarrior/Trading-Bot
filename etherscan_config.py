"""Shared helpers for configuring Etherscan API access.

This module centralises the logic for loading and rotating Etherscan API
keys.  Both the trading bot and auxiliary utilities (like the manual wallet
checker) rely on this behaviour so we keep it in a single place to avoid
subtle drift in default credentials or rotation semantics.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Sequence

# The production bot ships with a bundled set of fallback API keys.  Keeping
# the literal string in a single module avoids the different entry points from
# diverging when we update the bundled keys in the future.
_DEFAULT_ETHERSCAN_KEYS = (
    "HG9G9P667CSWMBM63XUWQQK4QERI49G2MI,"
    "DE19NIK8XYRV8BMZRYN6A5I8WNHZB3351Y,"
    "ADTS5TR8AXUNT8KSJYQXM6GM932SRYRDTW,"
    "132ZRTS6RAXQ3FMAF2ZB68QUQ4PEBCPM79,"
    "BA91JKSFWANGCE2B7X8DW4JEAFXD4HM294,"
    "Q6ZGWR2Q5M9Y4CCAZADJKKKJ7EUPKNZ7Q1"
)

_DEFAULT_ETHERSCAN_URLS = (
    "https://api.etherscan.io/v2/api",
    "https://api.etherscan.io/api",
)


def _parse_keys(raw: str) -> List[str]:
    """Split a comma-separated string into a list of API keys."""

    return [key.strip() for key in raw.split(",") if key and key.strip()]


def load_etherscan_keys(overrides: Sequence[str] | None = None) -> List[str]:
    """Return the configured Etherscan API keys.

    Priority order:
    1. Explicit overrides (from CLI flags, tests, etc.).
    2. ``ETHERSCAN_API_KEYS`` environment variable.
    3. ``ETHERSCAN_API_KEY`` environment variable.
    4. Bundled fallback keys used by ``ethereumbotv2``.
    """

    if overrides:
        return _parse_keys(",".join(overrides))

    raw = os.getenv("ETHERSCAN_API_KEYS")
    if not raw:
        raw = os.getenv("ETHERSCAN_API_KEY", _DEFAULT_ETHERSCAN_KEYS)
    return _parse_keys(raw or "")


def load_etherscan_base_urls(overrides: Sequence[str] | None = None) -> List[str]:
    """Return the configured Etherscan API base URLs in priority order."""

    if overrides:
        return _parse_keys(",".join(overrides))

    raw = os.getenv("ETHERSCAN_API_URLS")
    if raw:
        return _parse_keys(raw)
    return list(_DEFAULT_ETHERSCAN_URLS)


def make_key_rotator(keys: Sequence[str]) -> Iterator[str]:
    """Return an iterator cycling through the supplied keys indefinitely."""

    if not keys:
        # Yield empty strings forever to keep call sites simple – they can use
        # the value to detect the missing configuration and degrade gracefully.
        while True:
            yield ""

    while True:
        for key in keys:
            yield key


def make_key_getter(keys: Sequence[str]):
    """Return a callable that cycles through the supplied Etherscan keys."""

    rotator = make_key_rotator(keys)

    def _next_key() -> str:
        return next(rotator)

    return _next_key


__all__ = [
    "load_etherscan_keys",
    "load_etherscan_base_urls",
    "make_key_getter",
    "make_key_rotator",
]

