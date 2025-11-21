"""Utility helpers for discovering project social links from alternative sources.

The module queries lightweight HTTP endpoints (Etherscan token metadata,
GitHub READMEs, or configured search APIs) and normalises any Telegram,
X/Twitter, or general website URLs it finds. Results are cached to avoid
repeated lookups and rate-limit responses pause further calls briefly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from etherscan_config import load_etherscan_base_urls, load_etherscan_keys, make_key_getter

logger = logging.getLogger(__name__)

SOCIAL_CACHE: Dict[str, Tuple[float, List[str], Optional[str]]] = {}
SOCIAL_CACHE_TTL = int(os.getenv("SOCIAL_CACHE_TTL", "300") or "0")
SOCIAL_RATE_LIMIT_COOLDOWN = int(os.getenv("SOCIAL_RATE_LIMIT_COOLDOWN", "30") or "0")
SOCIAL_FETCH_TIMEOUT = int(os.getenv("SOCIAL_FETCH_TIMEOUT", "15") or "0")
ETHERSCAN_METADATA_URL = os.getenv("ETHERSCAN_METADATA_URL", "https://api.etherscan.io/api")
ETHERSCAN_METADATA_ACTION = os.getenv("ETHERSCAN_METADATA_ACTION", "tokeninfo")
ETHERSCAN_SOURCE_ACTION = os.getenv("ETHERSCAN_SOURCE_ACTION", "getsourcecode")
LIGHTWEIGHT_SEARCH_URL = os.getenv("SOCIAL_SEARCH_URL", "")
ETHERSCAN_CHAIN_ID = os.getenv("ETHERSCAN_CHAIN_ID", "1")

ETHERSCAN_BASE_URLS = [u for u in load_etherscan_base_urls() if u]
_next_etherscan_key = make_key_getter(load_etherscan_keys())

_TELEGRAM_RE = re.compile(r"https?://t(?:elegram)?\.me/[A-Za-z0-9_/-]+", re.IGNORECASE)
_TWITTER_RE = re.compile(r"https?://(?:twitter|x)\.com/[A-Za-z0-9_/-]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+", re.IGNORECASE)

_last_rate_limit_ts: float = 0.0


def _create_session(**kwargs) -> aiohttp.ClientSession:
    if "trust_env" not in kwargs:
        kwargs["trust_env"] = True
    if "timeout" not in kwargs:
        kwargs["timeout"] = aiohttp.ClientTimeout(total=SOCIAL_FETCH_TIMEOUT)
    return aiohttp.ClientSession(**kwargs)


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"https://{url}"
    return url


def _dedupe_links(links: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for link in links:
        norm = _normalize_url(link)
        if not norm or norm.lower() in seen:
            continue
        seen.add(norm.lower())
        ordered.append(norm)
    return ordered


def _maybe_apply_chainid(params: dict, base_url: str) -> dict:
    if "v2" in base_url and "chainid" not in params:
        params = {**params, "chainid": ETHERSCAN_CHAIN_ID}
    return params


def _extract_links_from_text(text: str) -> List[str]:
    links = []
    for regex in (_TELEGRAM_RE, _TWITTER_RE, _URL_RE):
        links.extend(regex.findall(text or ""))
    return _dedupe_links(links)


def _extract_links_from_entry(entry: dict) -> List[str]:
    links: List[str] = []
    for key in ("telegram", "twitter", "website", "websiteurl", "officialsite", "links", "github"):
        value = entry.get(key)
        if isinstance(value, str):
            links.append(value)
        elif isinstance(value, list):
            links.extend([v for v in value if isinstance(v, str)])
    for key, value in entry.items():
        if isinstance(value, str) and any(term in key.lower() for term in ("telegram", "twitter", "x", "website")):
            links.append(value)
    text_fragments = [v for v in entry.values() if isinstance(v, str)]
    for fragment in text_fragments:
        links.extend(_extract_links_from_text(fragment))
    return _dedupe_links(links)


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    global _last_rate_limit_ts
    try:
        async with session.get(url, params=params) as resp:
            if resp.status == 429:
                _last_rate_limit_ts = time.time()
                return None, "rate_limited"
            resp.raise_for_status()
            return await resp.json(), None
    except aiohttp.ClientResponseError as exc:
        if exc.status == 429:
            _last_rate_limit_ts = time.time()
            return None, "rate_limited"
        return None, f"http_{exc.status}"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return None, "network_error"
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"social discovery json fetch error: {exc}")
        return None, "unexpected_error"


async def _fetch_text(session: aiohttp.ClientSession, url: str) -> Tuple[Optional[str], Optional[str]]:
    global _last_rate_limit_ts
    try:
        async with session.get(url) as resp:
            if resp.status == 429:
                _last_rate_limit_ts = time.time()
                return None, "rate_limited"
            resp.raise_for_status()
            return await resp.text(), None
    except aiohttp.ClientResponseError as exc:
        if exc.status == 429:
            _last_rate_limit_ts = time.time()
            return None, "rate_limited"
        return None, f"http_{exc.status}"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return None, "network_error"
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"social discovery text fetch error: {exc}")
        return None, "unexpected_error"


async def _fetch_etherscan_metadata(session: aiohttp.ClientSession, token_addr: str) -> Tuple[List[dict], Optional[str]]:
    reasons: List[Optional[str]] = []

    for base_url in ETHERSCAN_BASE_URLS:
        api_key = _next_etherscan_key()
        if not api_key:
            reasons.append("missing_api_key")
            continue

        params = _maybe_apply_chainid(
            {
                "module": "token",
                "action": ETHERSCAN_METADATA_ACTION,
                "contractaddress": token_addr,
                "apikey": api_key,
            },
            base_url,
        )
        payload, reason = await _fetch_json(session, base_url, params=params)
        if not payload:
            reasons.append(reason)
            if reason == "rate_limited":
                _last_rate_limit_ts = time.time()
                break
            continue

        if str(payload.get("status")) != "1":
            message = str(payload.get("message") or "").lower()
            if "rate" in message:
                _last_rate_limit_ts = time.time()
                return [], "rate_limited"
            return [], "not_listed"

        result = payload.get("result")
        if isinstance(result, list):
            return [r for r in result if isinstance(r, dict)], None
        return [], None

    for reason in reasons:
        if reason:
            return [], reason
    return [], None


async def _fetch_github_socials_from_links(
    session: aiohttp.ClientSession, links: List[str]
) -> List[str]:
    socials: List[str] = []
    github_links = [link for link in links if "github.com" in link]
    for gh in github_links:
        parsed = urlparse(gh)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            continue
        owner, repo = parts[:2]
        for branch in ("main", "master"):
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
            text, reason = await _fetch_text(session, raw_url)
            if reason == "rate_limited":
                return socials
            if text:
                socials.extend(_extract_links_from_text(text))
                break
    return _dedupe_links(socials)


async def _fetch_contract_source_socials(
    session: aiohttp.ClientSession, token_addr: str
) -> Tuple[List[str], Optional[str]]:
    reasons: List[Optional[str]] = []

    for base_url in ETHERSCAN_BASE_URLS:
        api_key = _next_etherscan_key()
        if not api_key:
            reasons.append("missing_api_key")
            continue

        params = _maybe_apply_chainid(
            {
                "module": "contract",
                "action": ETHERSCAN_SOURCE_ACTION,
                "address": token_addr,
                "apikey": api_key,
            },
            base_url,
        )

        payload, reason = await _fetch_json(session, base_url, params=params)
        if not payload:
            reasons.append(reason)
            if reason == "rate_limited":
                break
            continue

        if str(payload.get("status")) != "1":
            message = str(payload.get("message") or "").lower()
            if "rate" in message:
                return [], "rate_limited"
            continue

        entries = payload.get("result")
        if not isinstance(entries, list):
            continue

        links: List[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for field in ("SourceCode", "ContractName", "CompilerVersion", "LicenseType"):
                value = entry.get(field)
                if isinstance(value, str):
                    links.extend(_extract_links_from_text(value))
        return _dedupe_links(links), None

    for reason in reasons:
        if reason:
            return [], reason
    return [], None


async def _perform_web_search(
    session: aiohttp.ClientSession, token_addr: str, pair_addr: str
) -> List[str]:
    if not LIGHTWEIGHT_SEARCH_URL:
        return []
    q = token_addr or pair_addr
    if not q:
        return []
    payload, reason = await _fetch_json(session, LIGHTWEIGHT_SEARCH_URL, params={"q": q})
    if not payload or reason:
        return []
    links: List[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("links"), list):
            links.extend([v for v in payload["links"] if isinstance(v, str)])
        if isinstance(payload.get("description"), str):
            links.extend(_extract_links_from_text(payload["description"]))
    return _dedupe_links(links)


async def fetch_social_links_async(
    token_addr: str, pair_addr: str = ""
) -> Tuple[List[str], Optional[str]]:
    """Return discovered social links for the token or pair address."""

    now = time.time()
    key = (token_addr or pair_addr or "").lower()
    if not key:
        return [], None

    if _last_rate_limit_ts and now - _last_rate_limit_ts < SOCIAL_RATE_LIMIT_COOLDOWN:
        return [], "rate_limited"

    cached = SOCIAL_CACHE.get(key)
    if cached and now - cached[0] < SOCIAL_CACHE_TTL:
        return cached[1], cached[2]

    links: List[str] = []
    reason: Optional[str] = None

    target_addr = token_addr or pair_addr

    async with _create_session() as session:
        meta_entries, meta_reason = await _fetch_etherscan_metadata(session, target_addr)
        reason = reason or meta_reason
        for entry in meta_entries:
            links.extend(_extract_links_from_entry(entry))
        links = _dedupe_links(links)

        if meta_entries:
            gh_links = await _fetch_github_socials_from_links(session, links)
            links = _dedupe_links([*links, *gh_links])

        if not links:
            source_links, source_reason = await _fetch_contract_source_socials(session, target_addr)
            reason = reason or source_reason
            if source_links:
                links = _dedupe_links([*links, *source_links])

        search_links = await _perform_web_search(session, token_addr, pair_addr)
        if search_links:
            links = _dedupe_links([*links, *search_links])

    SOCIAL_CACHE[key] = (now, links, reason)
    return links, reason


__all__ = [
    "fetch_social_links_async",
    "_extract_links_from_entry",
    "_extract_links_from_text",
    "_fetch_contract_source_socials",
]
