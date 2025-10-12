#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced CryptoBot w/ Honeypot + Renounce + Passing Refresh + Silent Checks + Advanced Contract Verification
------------------------------------------------------------------------------------------------------------
Features:
1) Paper trading (inactive).
2) Honeypot check on both token0 & token1 (via Honeypot.is).
3) If DexScreener liquidity ≤ $1 => fail as "Rugpull".
4) Re-check failing pairs using RECHECK_DELAYS.
5) Passing pairs get scheduled REFRESH checks (30s up to 15m).
6) Passing pairs also do SILENT checks every 10m for 2h to watch MC growth.
7) 1-minute “quick honeypot check” for passing pairs => if dev toggles honeypot.
8) Optionally check for “renounce” => if not renounced, fail or penalize.
9) Minimizes repeated "No more scheduled attempts" spam.
10) **NEW**: Advanced Contract Verification system (fetch from Etherscan, parse for high-risk patterns).
"""

import os
import time
import json
import logging
import re
from datetime import datetime, timedelta
try:
    from solidity_parser import parser
except ImportError:  # pragma: no cover - optional dependency
    parser = None
import contextlib
import io
from typing import Optional, Dict, Tuple, List
from openpyxl import Workbook, load_workbook
import traceback
import requests
import numpy as np

# Web3 & dependencies
from web3 import Web3, HTTPProvider
from web3.types import HexBytes
from eth_abi.abi import decode
from eth_utils import to_checksum_address

# Requests
import asyncio
import aiohttp
import threading
from concurrent.futures import Future
# Ensure local imports work even when script executed from a different directory
import sys
from pathlib import Path

# Add the script directory to sys.path for relative imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import wallet tracker with fallback for legacy filename
try:
    from wallet_tracker import (
        SmartWalletTracker,
        wallet_activity_callback,
        get_shared_tracker,
        set_notifier,
        set_etherscan_lookup_enabled as set_tracker_etherscan_enabled,
    )
except ModuleNotFoundError:
    try:  # pragma: no cover - backward compatibility
        from wallet_tracker_system import (
            SmartWalletTracker,
            wallet_activity_callback,
            get_shared_tracker,
            set_notifier,
            set_etherscan_lookup_enabled as set_tracker_etherscan_enabled,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wallet_tracker module not found; ensure wallet_tracker.py is present"
        ) from exc

# Fallback no-op to preserve compatibility if tracker module lacks the helper.
try:
    set_tracker_etherscan_enabled
except NameError:  # pragma: no cover - defensive
    def set_tracker_etherscan_enabled(enabled: bool, reason: str = "") -> None:
        return None

# On Windows the default ProactorEventLoop can emit "Event loop is closed"
# messages when asyncio.run() is used repeatedly. Switching to the
# Selector policy avoids those spurious warnings.
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

###########################################################
# 1. GLOBAL CONFIG & CONSTANTS
###########################################################

INFURA_URL = os.getenv(
    "INFURA_URL", "https://mainnet.infura.io/v3/92702160421d44dd8551754e78633549"
)
INFURA_URL_V3 = os.getenv(
    "INFURA_URL_V3", "https://mainnet.infura.io/v3/7236b8ffcce1451caaf08b275fb6dfc7"
)
# backup provider for rate limit situations
INFURA_URL_BACKUP = os.getenv(
    "INFURA_URL_BACKUP",
    "https://mainnet.infura.io/v3/ec4f1dd756644dcb9eb46762c4c4c9c0",
)
INFURA_URL_V3_BACKUP = os.getenv("INFURA_URL_V3_BACKUP", INFURA_URL_BACKUP)
ALCHEMY_URL = os.getenv(
    "ALCHEMY_URL",
    "https://eth-mainnet.g.alchemy.com/v2/ICzV00BkkR9g70gaOJrx0O80fO_c2oPB",
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8274484247:AAEoiTgXb6xLDmmSU3yLbqQaMOW81v541pY"
)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-4934038934")
# Pre-compute base URL to avoid repetition
TELEGRAM_BASE_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else ""
)

# The Graph configuration
GRAPH_URL = "https://gateway.thegraph.com/api/subgraphs/id/EYCKATKGBKLWvSfwvBjzfCBmGwYNdVkduYXVivCsLR"
GRAPH_BEARER = "6ab18515ae540220006db77a4472de7a"

UNISWAP_V2_FACTORY_ADDRESS = to_checksum_address(
    "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"
)
UNISWAP_V3_FACTORY_ADDRESS = to_checksum_address(
    "0x1F98431c8aD98523631AE4a59f267346ea31F984"
)

PAIR_CREATED_TOPIC_V2 = (
    "0x0d3648bd0f6ba80134a33ba9275ac585d9d315f0ad8355cddefde31afa28d0e9"
)
POOL_CREATED_TOPIC_V3 = (
    "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"
)

# backwards compatibility
UNISWAP_FACTORY_ADDRESS = UNISWAP_V2_FACTORY_ADDRESS
PAIR_CREATED_TOPIC = PAIR_CREATED_TOPIC_V2

MIN_PASS_THRESHOLD = 6

# Criteria thresholds
MIN_LIQUIDITY_USD = 20_000
MIN_VOLUME_USD = 25_000
MIN_FDV_USD = 40_000
MIN_MARKETCAP_USD = 40_000
MIN_BUYS_FIRST_HOUR = 20
MIN_TRADES_REQUIRED = 20

# Gem detection thresholds to reduce noise
MIN_GEM_MARKETCAP_USD = 50_000
MAX_GEM_MARKETCAP_USD = 200_000
MIN_GEM_LIQUIDITY_USD = 30_000
MIN_GEM_UNIQUE_BUYERS = 20
MAX_GEM_RISK_SCORE = 50
MIN_GEM_MARKETING_SCORE = 20
GEM_ALERT_SCORE = 60
GEM_WARNING_RISK_DELTA = 10
GEM_WARNING_MARKETING_DELTA = 10

# Wallet tracker settings
WALLET_REPORT_TTL = 600  # 10 minutes
WALLET_REPORT_CACHE: Dict[str, dict] = {}
wallet_tracker = None  # initialized after Web3 setup
wallet_monitor_tasks: Dict[str, Future] = {}
wallet_monitor_stops: Dict[str, asyncio.Event] = {}
MAX_WALLET_MONITORS = 5
wallet_event_loop = asyncio.new_event_loop()
threading.Thread(target=wallet_event_loop.run_forever, daemon=True).start()

import threading as _threading


# Known profitable wallets for smart money detection
SMART_MONEY_WALLETS = {
    "0x742d35cc6634c0532925a3b844bc454e4438f44e",
    "0xfe9e8709d3215310075d67e3ed32a380ccf451c8",
}

# Track overall market regime to tune detection thresholds
def detect_market_mode() -> str:
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        change = resp.json()["data"]["market_cap_change_percentage_24h_usd"]
        return "bull" if change and change > 0 else "bear"
    except Exception:
        return "bull"

MARKET_MODE = detect_market_mode()

def start_market_mode_monitor():
    def _run():
        global MARKET_MODE
        while True:
            MARKET_MODE = detect_market_mode()
            time.sleep(3600)
    threading.Thread(target=_run, daemon=True).start()

# Common constants
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# Tokens considered base assets (ignored for certain checks)
BASE_TOKENS = {
    "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
    "0xA0b86991C6218B36C1d19D4A2E9Eb0CE3606eB48",  # USDC
    "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
    ZERO_ADDRESS,
}
# WETH token address used to identify main token
WETH_ADDRESS = to_checksum_address("0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2")
# EIP-1967 implementation slot for proxies
EIP1967_IMPL_SLOT = bytes.fromhex(
    "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
)

# Uniswap pair ABI fragment
PAIR_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    }
]

# Burn event topic for liquidity removal
BURN_TOPIC = Web3.keccak(text="Burn(address,uint256,uint256,address)").hex()
# Swap event topics used for first-sell detection
SWAP_TOPIC_V2 = Web3.keccak(
    text="Swap(address,uint256,uint256,uint256,uint256,address)"
).hex()
SWAP_TOPIC_V3 = Web3.keccak(
    text="Swap(address,address,int256,int256,uint160,uint128,int24)"
).hex()


def get_non_weth_token(token0: str, token1: str) -> str:
    """Return the token address that is not WETH."""
    if token0.lower() == WETH_ADDRESS.lower():
        return token1
    return token0

# Re-check intervals for failing pairs
RECHECK_DELAYS = [60, 180, 300, 600, 1800]  # 1m,3m,5m,10m,30m

# Passing refresh intervals
PASSING_REFRESH_DELAYS = [
    30,
    45,
    45,
    30,
    60,
    300,
    600,
    900,
]  # 30s,45s,45s,30s,1m,5m,10m,15m

# Additional silent MC check
SILENT_CHECK_INTERVAL = 600  # 10 min
SILENT_CHECK_DURATION = 7200  # 2 hours

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAST_BLOCK_FILE_V2 = os.path.join(SCRIPT_DIR, "last_block_v2.json")
LAST_BLOCK_FILE_V3 = os.path.join(SCRIPT_DIR, "last_block_v3.json")
EXCEL_FILE = os.path.join(SCRIPT_DIR, "pairs.xlsx")
MAIN_LOOP_SLEEP = 2

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def init_excel(path: str):
    if not os.path.exists(path):
        wb = Workbook()
        ws = wb.active
        ws.title = "history"
        ws.append(
            [
                "pair",
                "token0",
                "token1",
                "detected_at",
                "passes",
                "total",
                "liquidity",
                "volume",
                "marketcap",
                "first_sell_address",
                "first_sell_block",
            ]
        )
        wb.save(path)


init_excel(EXCEL_FILE)


###########################################################
# 2. WEB3 & SESSION
###########################################################

w3_event = Web3(HTTPProvider(INFURA_URL))
w3_event_v3 = Web3(HTTPProvider(INFURA_URL_V3))
w3_backup_event = Web3(HTTPProvider(INFURA_URL_BACKUP))
w3_backup_event_v3 = Web3(HTTPProvider(INFURA_URL_V3_BACKUP))
w3_read = Web3(HTTPProvider(ALCHEMY_URL))

if not w3_event.is_connected():
    logger.warning("Event provider not connected => using backup provider")
    w3_event = w3_backup_event
if not w3_event.is_connected():
    logger.warning("Backup event provider not connected => fallback to w3_read")
    w3_event = w3_read
if not w3_event_v3.is_connected():
    logger.warning("V3 provider not connected => using backup provider")
    w3_event_v3 = w3_backup_event_v3
if not w3_event_v3.is_connected():
    logger.warning("Backup v3 provider not connected => fallback to w3_read")
    w3_event_v3 = w3_read

FETCH_TIMEOUT = 30


def safe_block_number(is_v3: bool = False) -> int:
    """Return latest block number with automatic provider fallback."""
    global w3_event, w3_event_v3
    provider = w3_event_v3 if is_v3 else w3_event
    backup_url = INFURA_URL_V3_BACKUP if is_v3 else INFURA_URL_BACKUP
    try:
        return provider.eth.block_number
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate" in msg:
            logger.warning(
                f"Rate limit on {'v3' if is_v3 else 'v2'} provider => switching to backup"
            )
            new_provider = Web3(HTTPProvider(backup_url))
            if is_v3:
                w3_event_v3 = new_provider
            else:
                w3_event = new_provider
            try:
                return new_provider.eth.block_number
            except Exception as e2:
                logger.error(f"backup block number error: {e2}")
        logger.error(f"block number error: {e}")
        return w3_read.eth.block_number


def safe_get_logs(filter_params: dict, is_v3: bool = False) -> List[dict]:
    """Get logs with fallback to backup provider on rate limit."""
    global w3_event, w3_event_v3
    provider = w3_event_v3 if is_v3 else w3_event
    backup_url = INFURA_URL_V3_BACKUP if is_v3 else INFURA_URL_BACKUP
    try:
        return provider.eth.get_logs(filter_params)
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate" in msg:
            logger.warning(
                f"Rate limit on {'v3' if is_v3 else 'v2'} get_logs => switching to backup"
            )
            new_provider = Web3(HTTPProvider(backup_url))
            if is_v3:
                w3_event_v3 = new_provider
            else:
                w3_event = new_provider
            try:
                return new_provider.eth.get_logs(filter_params)
            except Exception as e2:
                logger.error(f"backup get_logs error: {e2}")
        logger.error(f"get_logs {'v3' if is_v3 else 'v2'} error: {e}")
        time.sleep(5)
        return []



###########################################################
# 3. TELEGRAM
###########################################################

async def async_send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to Telegram asynchronously."""
    if not TELEGRAM_BASE_URL or not TELEGRAM_CHAT_ID:
        return False
    url = f"{TELEGRAM_BASE_URL}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, timeout=FETCH_TIMEOUT) as resp:
                payload = await resp.json()
                return bool(payload.get("ok"))
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False

def send_telegram_message(text: str, parse_mode: str = "HTML") -> bool:
    """Safe to call from inside or outside an event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_send_telegram_message(text, parse_mode))
    else:
        loop.create_task(async_send_telegram_message(text, parse_mode))
        return True

# Wire wallet tracker notifications to Telegram (Patch A)
try:
    set_notifier(send_telegram_message, async_send_telegram_message)
except Exception:
    pass




# ---------------------------------------------------------
# Telegram command listener (polling): /monitor_on, /monitor_off, /monitor_off_all, /monitor_list
# ---------------------------------------------------------
def _telegram_command_listener():
    if not TELEGRAM_BASE_URL or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram command listener disabled (no credentials)")
        return
    offset = 0
    import time as _time, requests
    session = requests.Session()
    while True:
        try:
            resp = session.get(
                f"{TELEGRAM_BASE_URL}/getUpdates",
                params={"timeout": 25, "offset": offset},
                timeout=FETCH_TIMEOUT + 5,
            )
            data = resp.json() if resp.ok else {}
            for upd in data.get("result", []):
                offset = upd.get("update_id", 0) + 1
                msg = upd.get("message") or upd.get("edited_message") or {}
                chat_id = str(msg.get("chat", {}).get("id", ""))
                # Restrict to configured chat unless left blank
                if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
                    continue
                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue
                parts = text.split()
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                if cmd in ("/monitor_on", "/monitoron"):
                    token = _resolve_main_token_from_arg(arg)
                    if not token:
                        send_telegram_message("Usage: /monitor_on <token-or-pair-address>")
                        continue
                    start_wallet_monitor(token)
                    send_telegram_message(f"✅ Monitoring started for <code>{token}</code>")
                elif cmd in ("/monitor_off", "/monitoroff"):
                    if not arg:
                        send_telegram_message("Usage: /monitor_off <token-or-pair-address>")
                        continue
                    ok = stop_wallet_monitor(arg)
                    if ok:
                        send_telegram_message(f"🛑 Monitoring stopped for <code>{arg}</code>")
                    else:
                        send_telegram_message(f"ℹ️ Not monitoring <code>{arg}</code>")
                elif cmd in ("/monitor_off_all", "/monitoroffall"):
                    n = stop_all_wallet_monitors()
                    send_telegram_message(f"🧹 Stopped {n} wallet monitor(s).")
                elif cmd in ("/monitor_list", "/monitorlist"):
                    lst = list_wallet_monitors()
                    if not lst:
                        send_telegram_message("No active wallet monitors.")
                    else:
                        body = "\n".join(f"- <code>{x}</code>" for x in lst)
                        send_telegram_message(f"Active monitors ({len(lst)}):\n{body}")
        except Exception as e:
            logger.debug(f"telegram listener error: {e}")
        finally:
            _time.sleep(1)


# Start the Telegram command listener in a background thread
_threading.Thread(target=_telegram_command_listener, daemon=True).start()

###########################################################
# 4. FAIL BUFFER
###########################################################

FAILED_RECHECKS_BUFFER: List[str] = []


def add_failed_recheck_message(msg: str):
    FAILED_RECHECKS_BUFFER.append(msg)


def maybe_flush_failed_rechecks():
    if FAILED_RECHECKS_BUFFER:
        log_msg = "[FailedRechecks]\n" + "\n".join(FAILED_RECHECKS_BUFFER)
        logger.info(log_msg)
    FAILED_RECHECKS_BUFFER.clear()


###########################################################
# 5. HONEYPOT + RENOUNCE
###########################################################

HONEYPOT_API_URL = "https://api.honeypot.is/v2/IsHoneypot"


def check_honeypot_is(
    token_addr: str, pair_addr: str = None, chain_id: int = None
) -> bool:
    """Return True if Honeypot.is flags the token as a honeypot."""

    async def _check() -> bool:
        params = {"address": token_addr}
        if chain_id is not None:
            params["chainID"] = chain_id
        if pair_addr:
            params["pair"] = pair_addr

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(HONEYPOT_API_URL, params=params, timeout=15) as resp:
                    j = await resp.json()
            hp = j.get("honeypotResult", {}).get("isHoneypot")
            if hp is True:
                return True
            if hp is False:
                return False
        except Exception as e:
            logger.debug(f"honeypot api error: {e}")

        try:
            url = f"https://honeypot.is/ethereum?address={token_addr}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as r:
                    text = await r.text()
            low = text.lower()
            if "honeypot detected" in low or "this token appears to be a honeypot" in low:
                return True
        except Exception as e:
            logger.debug(f"honeypot html error: {e}")
        return False

    return asyncio.run(_check())


def check_is_renounced(token_addr: str) -> bool:
    """Check if ownership has been renounced using multiple heuristics."""
    owner_addr, _, _ = get_owner_info(token_addr)
    if owner_addr and owner_addr.lower() == ZERO_ADDRESS.lower():
        return True
    if check_renounced_by_event(token_addr):
        return True
    return False


###########################################################
# 8. DEXSCREENER
###########################################################

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
DEXSCREENER_PAIR_URL = "https://api.dexscreener.com/latest/dex/pairs/ethereum"
DEXSCREENER_CACHE: Dict[str, Tuple[float, dict]] = {}
DEXSCREENER_CACHE_TTL = 300  # 5 minutes
DEXSCREENER_PAIR_TTL = 10  # short TTL for pair endpoint


async def _check_liquidity_locked_etherscan_async(pair_addr: str) -> bool:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return False
    api_key = get_next_etherscan_key()
    if not api_key:
        return False
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": pair_addr,
        "page": 1,
        "offset": 20,
        "sort": "asc",
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                j = await r.json()
        if j.get("status") != "1":
            return False
        for tx in j.get("result", []):
            if tx.get("from", "").lower() == pair_addr.lower():
                to_addr = tx.get("to", "").lower()
                if to_addr in {
                    ZERO_ADDRESS.lower(),
                    "0x000000000000000000000000000000000000dead",
                }:
                    return True
                info = fetch_contract_source_etherscan(to_addr)
                name = info.get("contractName", "").lower()
                if any(k in name for k in ["lock", "locker", "unicrypt", "team", "pink"]):
                    return True
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"lock lookup failed: {e}")
    except Exception as e:
        logger.debug(f"etherscan lock check error: {e}")
    return False

def check_liquidity_locked_etherscan(pair_addr: str) -> bool:
    return asyncio.run(_check_liquidity_locked_etherscan_async(pair_addr))


async def _fetch_dexscreener_data_async(token_addr: str, pair_addr: str) -> Optional[dict]:
    try:
        now = time.time()
        pair_info = None

        # --- token-based lookup (preferred) ---
        key = token_addr.lower()
        cached = DEXSCREENER_CACHE.get(key)
        if cached and now - cached[0] < DEXSCREENER_CACHE_TTL:
            jdata = cached[1]
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{DEXSCREENER_SEARCH_URL}?q={token_addr}",
                    timeout=FETCH_TIMEOUT,
                ) as resp:
                    jdata = await resp.json()
            DEXSCREENER_CACHE[key] = (now, jdata)

        pairs = jdata.get("pairs", []) if jdata else []
        for p in pairs:
            if p.get("pairAddress", "").lower() == pair_addr.lower():
                pair_info = p
                break

        # --- fallback to pair endpoint when token search fails ---
        if not pair_info:
            pkey = f"pair:{pair_addr.lower()}"
            cached_pair = DEXSCREENER_CACHE.get(pkey)
            if cached_pair and now - cached_pair[0] < DEXSCREENER_PAIR_TTL:
                pdata = cached_pair[1]
            else:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{DEXSCREENER_PAIR_URL}/{pair_addr}",
                        timeout=FETCH_TIMEOUT,
                    ) as resp:
                        pdata = await resp.json()
                DEXSCREENER_CACHE[pkey] = (now, pdata)

            plist = pdata.get("pairs", []) if pdata else []
            if plist:
                pair_info = plist[0]
            else:
                return None

        price_usd = float(pair_info.get("priceUsd", 0) or 0)
        liq_usd = float(pair_info.get("liquidity", {}).get("usd", 0) or 0)
        vol_24h = float(pair_info.get("volume", {}).get("h24", 0) or 0)
        fdv = float(pair_info.get("fdv", 0) or 0)
        mc = float(pair_info.get("marketCap", 0) or 0)

        tx24 = pair_info.get("txns", {}).get("h24", {})
        buys_24 = int(tx24.get("buys", 0))
        sells_24 = int(tx24.get("sells", 0))

        base_token = pair_info.get("baseToken", {})
        base_name = base_token.get("name", "").strip()
        info_section = pair_info.get("info", {})
        logo_url = info_section.get("imageUrl", "")
        websites = [w.get("url") for w in info_section.get("websites", []) if w.get("url")]
        socials = [s.get("url") for s in info_section.get("socials", []) if s.get("url")]
        # Determine locked liquidity via DexScreener label and Etherscan data
        locked = False
        labels = pair_info.get("labels", [])
        if "locked" in labels:
            locked = True
        else:
            locked = await _check_liquidity_locked_etherscan_async(pair_addr)

        # Detect paid promotions/trending on Dex platforms
        dex_paid = any(lbl.lower() in {"promoted", "boosted", "paid"} for lbl in labels)

        return {
            "priceUsd": price_usd,
            "liquidityUsd": liq_usd,
            "volume24h": vol_24h,
            "fdv": fdv,
            "marketCap": mc,
            "buys": buys_24,
            "sells": sells_24,
            "baseTokenName": base_name,
            "baseTokenLogo": logo_url,
            "socialLinks": websites + socials,
            "lockedLiquidity": locked,
            "dexPaid": dex_paid,
        }
    except Exception as e:
        logger.debug(f"fetch_dexscreener_data error: {e}")
        return None

def fetch_dexscreener_data(token_addr: str, pair_addr: str) -> Optional[dict]:
    return asyncio.run(_fetch_dexscreener_data_async(token_addr, pair_addr))


async def _check_recent_liquidity_removal_async(pair_addr: str, timeframe_sec: int = 600) -> bool:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return False
    api_key = get_next_etherscan_key()
    if not api_key:
        return False
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": pair_addr,
        "page": 1,
        "offset": 10,
        "sort": "desc",
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                j = await r.json()
        if j.get("status") != "1":
            return False
        now_ts = time.time()
        for tx in j.get("result", []):
            if tx.get("to", "").lower() == ZERO_ADDRESS.lower():
                ts = int(tx.get("timeStamp", "0"))
                if now_ts - ts <= timeframe_sec:
                    return True
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"liquidity removal lookup failed: {e}")
    except Exception as e:
        logger.debug(f"Etherscan removal check error: {e}")
    return False

def check_recent_liquidity_removal(pair_addr: str, timeframe_sec: int = 600) -> bool:
    return asyncio.run(_check_recent_liquidity_removal_async(pair_addr, timeframe_sec))


###########################################################
# 9. ADVANCED CONTRACT VERIFICATION
#    (Fetch from Etherscan, analyze code, detect high risk)
###########################################################

# Example: Set your Etherscan API keys if you want to do real contract checks
_raw_keys = os.getenv(
    "ETHERSCAN_API_KEYS",
    os.getenv(
        "ETHERSCAN_API_KEY",
        "ADTS5TR8AXUNT8KSJYQXM6GM932SRYRDTW,DE19NIK8XYRV8BMZRYN6A5I8WNHZB3351Y,QY8285FMHAY16N9Z721SVR27ZEP13DZS5",
    ),
)
ETHERSCAN_API_KEY_LIST = [k.strip() for k in _raw_keys.split(",") if k.strip()]
ETHERSCAN_API_INDEX = 0

def get_next_etherscan_key() -> str:
    global ETHERSCAN_API_INDEX
    if not ETHERSCAN_API_KEY_LIST:
        return ""
    key = ETHERSCAN_API_KEY_LIST[ETHERSCAN_API_INDEX]
    ETHERSCAN_API_INDEX = (ETHERSCAN_API_INDEX + 1) % len(ETHERSCAN_API_KEY_LIST)
    return key

ETHERSCAN_API_URL = "https://api.etherscan.io/api"


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


ETHERSCAN_LOOKUPS_ENABLED = _env_flag("ENABLE_ETHERSCAN_LOOKUPS", True)
ETHERSCAN_DISABLED_REASON: Optional[str] = None


def disable_etherscan_lookups(reason: str) -> None:
    global ETHERSCAN_LOOKUPS_ENABLED, ETHERSCAN_DISABLED_REASON
    if ETHERSCAN_LOOKUPS_ENABLED:
        ETHERSCAN_LOOKUPS_ENABLED = False
        ETHERSCAN_DISABLED_REASON = reason
        logger.warning("Disabling Etherscan lookups: %s", reason)
        try:
            set_tracker_etherscan_enabled(False, reason)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to propagate Etherscan disable to tracker", exc_info=True)


# Initialize wallet tracker now that helper functions are defined
wallet_tracker = get_shared_tracker(w3_read, get_next_etherscan_key)
start_market_mode_monitor()


class ContractVerificationStatus:
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    ERROR = "error"


async def _fetch_contract_source_etherscan_async(token_addr: str) -> dict:
    """Return verified source code information from Etherscan.

    The returned dictionary contains:
        ``status``: ``verified`` | ``unverified`` | ``error``
        ``source``: list of ``{"filename": str, "content": str}``
        ``compilerVersion``: compiler version string
        ``contractName``: contract name
    """
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return {
            "status": ContractVerificationStatus.ERROR,
            "source": [],
            "compilerVersion": "",
            "contractName": "",
        }
    token_addr = token_addr.lower()
    url = (
        f"{ETHERSCAN_API_URL}?module=contract&action=getsourcecode"
        f"&address={token_addr}&apikey={get_next_etherscan_key()}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as r:
                j = await r.json()
        status = j.get("status")
        result = j.get("result", [])

        if status == "1" and len(result) > 0:
            # Etherscan returns a list. If SourceCode is empty => unverified
            if not result[0].get("SourceCode"):
                return {
                    "status": ContractVerificationStatus.UNVERIFIED,
                    "source": [],
                    "compilerVersion": "",
                    "contractName": "",
                }
            else:
                # Verified
                scode = result[0].get("SourceCode", "")
                cname = result[0].get("ContractName", "")
                compiler = result[0].get("CompilerVersion", "")

                sources_list = []
                stripped = scode.strip()
                if stripped.startswith("{"):
                    try:
                        data = json.loads(scode)
                        for fname, info in data.get("sources", {}).items():
                            content = info.get("content", "")
                            sources_list.append({"filename": fname, "content": content})
                    except Exception:
                        sources_list.append(
                            {"filename": cname or "contract.sol", "content": scode}
                        )
                else:
                    sources_list.append(
                        {"filename": cname or "contract.sol", "content": scode}
                    )

                return {
                    "status": ContractVerificationStatus.VERIFIED,
                    "source": sources_list,
                    "compilerVersion": compiler,
                    "contractName": cname,
                }
        else:
            return {
                "status": ContractVerificationStatus.ERROR,
                "source": [],
                "compilerVersion": "",
                "contractName": "",
            }

    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"source fetch failed: {e}")
        return {
            "status": ContractVerificationStatus.ERROR,
            "source": [],
            "compilerVersion": "",
            "contractName": "",
        }
    except Exception as e:
        logger.warning(f"fetch_contract_source_etherscan error: {e}")
        return {
            "status": ContractVerificationStatus.ERROR,
            "source": [],
            "compilerVersion": "",
            "contractName": "",
        }

def fetch_contract_source_etherscan(token_addr: str) -> dict:
    return asyncio.run(_fetch_contract_source_etherscan_async(token_addr))


def get_contract_creator(token_addr: str) -> Optional[str]:
    """Return the deployer address via Etherscan as a fallback."""
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return None
    api_key = get_next_etherscan_key()
    if not api_key:
        return None
    url = (
        f"{ETHERSCAN_API_URL}?module=contract&action=getcontractcreation"
        f"&contractaddresses={token_addr}&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
        if data.get("status") == "1" and data.get("result"):
            return data["result"][0].get("contractCreator")
    except requests.RequestException as exc:
        disable_etherscan_lookups(f"contract creator lookup failed: {exc}")
    except Exception:
        logger.debug("contract creator fetch failed", exc_info=True)
    return None


def get_owner_info(token_addr: str):
    """Return (owner_or_creator, eth_balance, token_balance)."""
    owner_addr = None
    for fn in ["owner", "getOwner", "ownerAddress", "admin", "administrator"]:
        abi = [
            {
                "constant": True,
                "inputs": [],
                "name": fn,
                "outputs": [{"name": "", "type": "address"}],
                "type": "function",
            }
        ]
        try:
            c = w3_read.eth.contract(to_checksum_address(token_addr), abi=abi)
            owner_addr = getattr(c.functions, fn)().call()
            break
        except Exception:
            continue

    if not owner_addr:
        owner_addr = get_contract_creator(token_addr)

    if owner_addr:
        try:
            bal_eth = w3_read.eth.get_balance(owner_addr)
            token_abi = [
                {
                    "constant": True,
                    "inputs": [{"name": "", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function",
                }
            ]
            token = w3_read.eth.contract(to_checksum_address(token_addr), abi=token_abi)
            bal_token = token.functions.balanceOf(owner_addr).call()
            return owner_addr, w3_read.from_wei(bal_eth, "ether"), bal_token
        except Exception:
            pass
        try:
            bal_eth = w3_read.eth.get_balance(owner_addr)
            return owner_addr, w3_read.from_wei(bal_eth, "ether"), None
        except Exception:
            return owner_addr, None, None
    return None, None, None


async def _check_owner_wallet_activity_async(token_addr: str, owner_addr: str) -> bool:
    """Return True if suspicious owner transactions detected recently."""
    if not owner_addr:
        return False
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return False
    api_key = get_next_etherscan_key()
    if not api_key:
        return False
    params = {
        "module": "account",
        "action": "tokentx",
        "address": owner_addr,
        "contractaddress": token_addr,
        "page": 1,
        "offset": 10,
        "sort": "desc",
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                j = await r.json()
        if j.get("status") != "1":
            return False
        total_supply = None
        try:
            abi = [{"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}]
            c = w3_read.eth.contract(to_checksum_address(token_addr), abi=abi)
            total_supply = c.functions.totalSupply().call()
        except Exception:
            pass
        for tx in j.get("result", []):
            val = int(tx.get("value", "0"))
            if total_supply and val > total_supply * 0.05:
                return True
        return False
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"owner activity lookup failed: {e}")
        return False
    except Exception as e:
        logger.debug(f"owner activity check error: {e}")
        return False


def check_owner_wallet_activity(token_addr: str, owner_addr: str) -> bool:
    return asyncio.run(_check_owner_wallet_activity_async(token_addr, owner_addr))


async def _fetch_third_party_risk_score_async(token_addr: str) -> Optional[int]:
    url = (
        "https://api.gopluslabs.io/api/v1/token_security/1?contract_addresses="
        f"{token_addr}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=FETCH_TIMEOUT) as resp:
                data = await resp.json()
        entry = data.get("result", {}).get(token_addr.lower())
        if entry and "total_score" in entry:
            return int(entry["total_score"])
    except Exception as e:
        logger.debug(f"third-party score error: {e}")
    return None


def get_third_party_risk_score(token_addr: str) -> Optional[int]:
    return asyncio.run(_fetch_third_party_risk_score_async(token_addr))


def get_lp_total_supply(pair_addr: str) -> Optional[int]:
    """Return current totalSupply for the LP token."""
    try:
        c = w3_read.eth.contract(to_checksum_address(pair_addr), abi=PAIR_ABI)
        return c.functions.totalSupply().call()
    except Exception as e:
        logger.debug(f"lp supply fetch error: {e}")
        return None


def _node_contains_identifiers(node, keywords) -> bool:
    """Recursively search an AST node for identifiers containing keywords."""
    if node is None:
        return False
    if isinstance(node, dict):
        if node.get("type") == "Identifier":
            name = str(node.get("name", "")).lower()
            for kw in keywords:
                if kw in name:
                    return True
        for child in node.values():
            if _node_contains_identifiers(child, keywords):
                return True
    elif isinstance(node, list):
        for elem in node:
            if _node_contains_identifiers(elem, keywords):
                return True
    return False


def _iter_modifiers(node):
    if isinstance(node, dict):
        if node.get("type") == "ModifierDefinition":
            yield node
        for child in node.values():
            if isinstance(child, (dict, list)):
                yield from _iter_modifiers(child)
    elif isinstance(node, list):
        for elem in node:
            yield from _iter_modifiers(elem)


def analyze_solidity_source(source_text: str) -> dict:
    """
    Returns dictionary of risk flags & a total riskScore.
    Example:
      {
        "ownerFunctions": bool,
        "canSetFees": bool,
        "maxTaxPossible": "unbounded" or "some guess",
        "canBlacklist": bool,
        "canPauseTrading": bool,
        "upgradeableProxy": bool,
        "renounceOwnerImplemented": bool,
        "botWhitelist": bool,
        "score": int
      }
    """
    text_lower = source_text.lower()
    flags = {
        "ownerFunctions": False,
        "canSetFees": False,
        "maxTaxPossible": "unknown",
        "canBlacklist": False,
        "canPauseTrading": False,
        "upgradeableProxy": False,
        "renounceOwnerImplemented": False,
        "transferBlockingModifier": False,
        "botWhitelist": False,
        "canModifyLimits": False,
        "canMint": False,
        "ownerActivity": False,
        "walletDrainer": False,
        "delegatecall": False,
        "selfDestruct": False,
        "tokenomicsPatterns": False,
        "autoLiquidityAdd": False,
        "ownerPrivileges": False,
        "thirdPartyScore": None,
        "vestingOrTimelock": False,
        "privateSaleFunctions": False,
    }

    # parse AST to detect transfer/sell usage inside modifiers
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ast = parser.parse(source_text)
        for mod in _iter_modifiers(ast):
            body = mod.get("body")
            if _node_contains_identifiers(body, {"transfer", "sell"}):
                flags["transferBlockingModifier"] = True
                break
    except Exception:
        pass

    # 1) "onlyOwner" or "Ownable"
    if "onlyowner" in text_lower or "ownable" in text_lower:
        flags["ownerFunctions"] = True

    # 2) set fee or tax
    set_fee_pattern = re.compile(r"function\s+set\w*fee", re.IGNORECASE)
    if set_fee_pattern.search(source_text):
        flags["canSetFees"] = True
        # Possibly guess a max tax
        flags["maxTaxPossible"] = "potentially 100%"

    # 3) modify trading limits or tax
    if re.search(r"function\s+(setmaxwallet|setmaxtx|setmaxtransaction|updatetax|settax)", source_text, re.IGNORECASE):
        flags["canModifyLimits"] = True

    # 4) minting ability
    if re.search(r"function\s+mint", source_text, re.IGNORECASE):
        flags["canMint"] = True

    if re.search(r"liquidityfee|marketingfee|reflection", text_lower):
        flags["tokenomicsPatterns"] = True
    if re.search(r"addliquidity|swapandliquify|autoliquidity", text_lower):
        flags["autoLiquidityAdd"] = True
    if re.search(r"setowner\s*\(|transferownership\s*\(", text_lower):
        flags["ownerPrivileges"] = True
    if re.search(r"vesting|timelock", text_lower):
        flags["vestingOrTimelock"] = True
    if re.search(r"claim|unlock|release", text_lower):
        flags["privateSaleFunctions"] = True

    # 5) blacklisting
    if "blacklist(" in text_lower or "addtoblacklist(" in text_lower:
        flags["canBlacklist"] = True

    # 6) can pause trading
    if "enabletrading(" in text_lower or "settradingenabled(" in text_lower:
        flags["canPauseTrading"] = True

    # 7) upgradeable proxy
    if "proxy" in text_lower or "upgradeable" in text_lower:
        flags["upgradeableProxy"] = True

    # 8) renounceOwner
    if "function renounceownership" in text_lower:
        flags["renounceOwnerImplemented"] = True

    # 9) bot or whitelist detection
    if re.search(r"\b(bot\w*|whitelist\w*)", text_lower):
        flags["botWhitelist"] = True

    # 10) wallet drainer patterns
    wallet_drainer_regex = re.compile(
        r"\b(?:transfer|transferFrom|safeTransfer|safeTransferFrom)\s*\(\s*msg\.sender",
        re.IGNORECASE,
    )
    if wallet_drainer_regex.search(source_text):
        flags["walletDrainer"] = True
    if re.search(r"delegatecall\s*\(", source_text):
        flags["delegatecall"] = True
    if re.search(r"(selfdestruct|suicide)\s*\(", source_text, re.IGNORECASE):
        flags["selfDestruct"] = True

    # risk scoring
    risk_score = 0
    if flags["ownerFunctions"]:
        risk_score += 2
    if flags["canSetFees"]:
        risk_score += 2
    if flags["maxTaxPossible"] == "potentially 100%":
        risk_score += 3
    if flags["canBlacklist"]:
        risk_score += 3
    if flags["canPauseTrading"]:
        risk_score += 3
    if flags["upgradeableProxy"]:
        risk_score += 4
    if not flags["renounceOwnerImplemented"]:
        risk_score += 2
    if flags["transferBlockingModifier"]:
        risk_score += 4
    if flags["botWhitelist"]:
        risk_score += 3
    if flags["canModifyLimits"]:
        risk_score += 2
    if flags["canMint"]:
        risk_score += 3
    if flags["walletDrainer"]:
        risk_score += 5
    if flags["delegatecall"]:
        risk_score += 4
    if flags["selfDestruct"]:
        risk_score += 4
    if flags["tokenomicsPatterns"]:
        risk_score += 1
    if flags["autoLiquidityAdd"]:
        risk_score += 2
    if flags["ownerPrivileges"]:
        risk_score += 3
    if flags["privateSaleFunctions"]:
        risk_score += 1
    if flags["thirdPartyScore"] is not None:
        risk_score += max(0, (100 - flags["thirdPartyScore"]) // 20)

    flags["score"] = risk_score
    return flags


def detect_proxy_implementation(addr: str) -> Optional[str]:
    """Return implementation address if contract is a proxy (EIP1967)."""
    try:
        raw = w3_read.eth.get_storage_at(to_checksum_address(addr), EIP1967_IMPL_SLOT)
        if int.from_bytes(raw, "big") != 0:
            return to_checksum_address("0x" + raw.hex()[-40:])
    except Exception:
        pass
    return None


async def _check_renounced_by_event_async(addr: str) -> bool:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return False
    topic0 = Web3.keccak(text="OwnershipTransferred(address,address)").hex()
    zero_topic = "0x" + "0" * 64
    params = {
        "module": "logs",
        "action": "getLogs",
        "fromBlock": "0",
        "toBlock": "latest",
        "address": addr,
        "topic0": topic0,
        "topic2": zero_topic,
        "apikey": get_next_etherscan_key(),
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=20) as r:
                j = await r.json()
        if j.get("status") == "1" and j.get("result"):
            return True
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"renounce lookup failed: {e}")
    except Exception:
        pass
    return False


def check_renounced_by_event(addr: str) -> bool:
    return asyncio.run(_check_renounced_by_event_async(addr))


def run_slither_analysis(source: object) -> dict:
    """Run Slither on the given source code and return issue count.

    ``source`` can be a string (single Solidity file) or a list of
    ``{"filename": str, "content": str}`` dictionaries representing a project.
    """
    import tempfile, subprocess, json as _json, shutil

    result = {"slitherIssues": None}
    if shutil.which("slither") is None:
        logger.warning("slither executable not found; skipping analysis")
        result["slitherIssues"] = "not_installed"
        return result
    try:
        with tempfile.TemporaryDirectory() as tmpd:
            if isinstance(source, str):
                src_path = os.path.join(tmpd, "contract.sol")
                with open(src_path, "w", encoding="utf-8") as f:
                    f.write(source)
                target = src_path
            else:
                for part in source:
                    fpath = os.path.join(tmpd, part.get("filename", "contract.sol"))
                    os.makedirs(os.path.dirname(fpath), exist_ok=True)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(part.get("content", ""))
                target = tmpd

            out_json = os.path.join(tmpd, "slither.json")
            cmd = ["slither", target, "--json", out_json, "--solc-disable-warnings"]
            proc = subprocess.run(
                cmd,
                check=True,
                timeout=60,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            data = _json.load(open(out_json, encoding="utf-8"))
            issues = data.get("results", {}).get("detectors", [])
            result["slitherIssues"] = len(issues)
    except subprocess.CalledProcessError as e:
        stdout = (
            e.stdout.decode("utf-8", errors="replace")
            if isinstance(e.stdout, bytes)
            else str(e.stdout)
        )
        stderr = (
            e.stderr.decode("utf-8", errors="replace")
            if isinstance(e.stderr, bytes)
            else str(e.stderr)
        )
        logger.warning(f"slither analysis failed: {stdout}\n{stderr}")
    except Exception as e:
        logger.warning(f"slither analysis error: {e}")
    return result


def advanced_contract_check(token_addr: str) -> dict:
    """Fetch source and analyze risk. Handles proxies and renounce checks."""
    impl = detect_proxy_implementation(token_addr)
    target = impl or token_addr
    info = fetch_contract_source_etherscan(target)
    owner_addr, owner_bal, owner_token_bal = get_owner_info(target)
    suspicious_activity = False
    if owner_addr:
        suspicious_activity = check_owner_wallet_activity(target, owner_addr)
    renounced = False
    if owner_addr and owner_addr.lower() == ZERO_ADDRESS.lower():
        renounced = True
    else:
        renounced = check_renounced_by_event(target)
    third_score = get_third_party_risk_score(target)
    private_sale = detect_private_sale_indicators(target)
    onchain_metrics = fetch_onchain_metrics(target)

    if info["status"] == ContractVerificationStatus.VERIFIED:
        combined = "".join(part["content"] + "\n" for part in info["source"])
        flags = analyze_solidity_source(combined)
        slither_res = run_slither_analysis(info["source"])
        if slither_res.get("slitherIssues") is not None:
            flags["slitherIssues"] = slither_res["slitherIssues"]
        else:
            flags["slitherIssues"] = "error"
        if impl:
            flags["upgradeableProxy"] = True
        if renounced:
            flags["renounced"] = True
        else:
            flags["renounced"] = False
        flags["thirdPartyScore"] = third_score
        score = flags.get("score", 0)
        if third_score is not None:
            score += max(0, (100 - third_score) // 20)
        owner_flag = flags.get("ownerActivity") or suspicious_activity
        if owner_flag:
            flags["ownerActivity"] = True
            score += 3
        if isinstance(flags.get("slitherIssues"), int):
            score += min(flags["slitherIssues"], 5)
        if not renounced:
            score += 2
        if score >= 10:
            st = "HIGH_RISK"
        else:
            st = "OK"
        result = {
            "verified": True,
            "riskScore": score,
            "riskFlags": flags,
            "status": st,
            "owner": owner_addr,
            "ownerBalanceEth": owner_bal,
            "ownerTokenBalance": owner_token_bal,
            "implementation": impl,
            "renounced": renounced,
            "slitherIssues": flags.get("slitherIssues"),
            "ownerActivity": suspicious_activity,
            "privateSale": private_sale,
            "onChainMetrics": onchain_metrics,
        }
        return result
    elif info["status"] == ContractVerificationStatus.UNVERIFIED:
        return {
            "verified": False,
            "riskScore": 9999,
            "riskFlags": {"ownerActivity": suspicious_activity},
            "status": "UNVERIFIED",
            "owner": owner_addr,
            "ownerBalanceEth": owner_bal,
            "ownerTokenBalance": owner_token_bal,
            "implementation": impl,
            "renounced": renounced,
            "slitherIssues": None,
        }
    else:
        return {
            "verified": False,
            "riskScore": 9999,
            "riskFlags": {"ownerActivity": suspicious_activity},
            "status": "ERROR",
            "owner": owner_addr,
            "ownerBalanceEth": owner_bal,
            "ownerTokenBalance": owner_token_bal,
            "implementation": impl,
            "renounced": renounced,
            "slitherIssues": None,
        }


###########################################################
# Additional Metrics & Bull Season Helpers
###########################################################

async def _fetch_holder_distribution_async(token_addr: str, limit: int = 10) -> List[dict]:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return []
    api_key = get_next_etherscan_key()
    if not api_key:
        return []
    params = {
        "module": "token",
        "action": "tokenholderlist",
        "contractaddress": token_addr,
        "page": 1,
        "offset": limit,
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                data = await r.json()
        if isinstance(data, dict):
            result = data.get("result", [])
            if isinstance(result, list):
                return result
        return []
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"holder distribution lookup failed: {e}")
        return []
    except Exception as e:
        logger.debug(f"holder distribution error: {e}")
        return []


def fetch_holder_distribution(token_addr: str, limit: int = 10) -> List[dict]:
    return asyncio.run(_fetch_holder_distribution_async(token_addr, limit))


async def _analyze_transfer_history_async(token_addr: str, limit: int = 100) -> dict:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return {
            "uniqueBuyers": 0,
            "uniqueSellers": 0,
            "smartMoneyCount": 0,
        }
    api_key = get_next_etherscan_key()
    metrics = {
        "uniqueBuyers": 0,
        "uniqueSellers": 0,
        "smartMoneyCount": 0,
    }
    if not api_key:
        return metrics
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": token_addr,
        "page": 1,
        "offset": limit,
        "sort": "asc",
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                data = await r.json()
        if data.get("status") != "1":
            return metrics
        buyers = set()
        sellers = set()
        for tx in data.get("result", []):
            frm = tx.get("from", "").lower()
            to = tx.get("to", "").lower()
            if frm in {ZERO_ADDRESS.lower(), token_addr.lower()}:
                buyers.add(to)
            elif to in {ZERO_ADDRESS.lower(), token_addr.lower()}:
                sellers.add(frm)
            if frm in SMART_MONEY_WALLETS or to in SMART_MONEY_WALLETS:
                metrics["smartMoneyCount"] += 1
        metrics["uniqueBuyers"] = len(buyers)
        metrics["uniqueSellers"] = len(sellers)
        return metrics
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"transfer history lookup failed: {e}")
        return metrics
    except Exception as e:
        logger.debug(f"transfer history error: {e}")
        return metrics


def analyze_transfer_history(token_addr: str, limit: int = 100) -> dict:
    return asyncio.run(_analyze_transfer_history_async(token_addr, limit))


async def _detect_private_sale_async(token_addr: str) -> dict:
    if not ETHERSCAN_LOOKUPS_ENABLED:
        return {"hasPresale": False, "largeTransfers": []}
    api_key = get_next_etherscan_key()
    result = {"hasPresale": False, "largeTransfers": []}
    if not api_key:
        return result
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": token_addr,
        "page": 1,
        "offset": 20,
        "sort": "asc",
        "apikey": api_key,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params, timeout=FETCH_TIMEOUT) as r:
                data = await r.json()
        if data.get("status") != "1":
            return result
        total_supply = None
        try:
            abi = [{"constant": True, "inputs": [], "name": "totalSupply", "outputs": [{"name": "", "type": "uint256"}], "type": "function"}]
            c = w3_read.eth.contract(to_checksum_address(token_addr), abi=abi)
            total_supply = c.functions.totalSupply().call()
        except Exception:
            pass
        for tx in data.get("result", []):
            val = int(tx.get("value", "0"))
            if total_supply and val > total_supply * 0.02:
                result["hasPresale"] = True
                result["largeTransfers"].append(tx.get("to"))
        return result
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        disable_etherscan_lookups(f"private sale lookup failed: {e}")
        return result
    except Exception as e:
        logger.debug(f"private sale detect error: {e}")
        return result


def detect_private_sale_indicators(token_addr: str) -> dict:
    return asyncio.run(_detect_private_sale_async(token_addr))


def has_private_sale(token_addr: str) -> bool:
    info = detect_private_sale_indicators(token_addr)
    return info.get("hasPresale", False)


def fetch_onchain_metrics(token_addr: str) -> dict:
    holders = fetch_holder_distribution(token_addr)
    if not isinstance(holders, list):
        holders = []
    transfers = analyze_transfer_history(token_addr)
    holder_share = None
    total_supply = 0
    try:
        abi = [{
            "constant": True,
            "inputs": [],
            "name": "totalSupply",
            "outputs": [{"name": "", "type": "uint256"}],
            "type": "function",
        }]
        c = w3_read.eth.contract(to_checksum_address(token_addr), abi=abi)
        total_supply = c.functions.totalSupply().call()
    except Exception as e:
        logger.debug(f"totalSupply fetch error: {e}")

    top_balance = 0
    for h in holders:
        if isinstance(h, dict):
            bal = int(h.get("balance", "0"))
            top_balance += bal
    if total_supply > 0:
        holder_share = top_balance / total_supply
    ratio = None
    if transfers["uniqueSellers"] > 0:
        ratio = transfers["uniqueBuyers"] / transfers["uniqueSellers"]
    return {
        "holderConcentration": holder_share,
        "uniqueBuyerSellerRatio": ratio,
        "smartMoneyCount": transfers["smartMoneyCount"],
    }


def get_current_liquidity(token_addr: str, pair_addr: str) -> float:
    """Helper to fetch current liquidity for a pair.

    The previous implementation mistakenly queried DexScreener with the pair
    address for both parameters which always returned empty data. By accepting
    the token address separately we ensure the lookup matches the pair and
    provides meaningful liquidity figures."""
    ds = fetch_dexscreener_data(token_addr, pair_addr)
    if ds:
        return ds.get("liquidityUsd", 0)
    return 0


def get_wallet_report(token_addr: str) -> dict:
    """Return cached wallet report or generate a new one."""
    key = token_addr.lower()
    now = time.time()
    data = WALLET_REPORT_CACHE.get(key)
    if data and now - data.get("ts", 0) < WALLET_REPORT_TTL:
        return data["report"]
    try:
        fut = asyncio.run_coroutine_threadsafe(
            wallet_tracker.generate_wallet_report(token_addr), wallet_event_loop
        )
        report = fut.result()
        WALLET_REPORT_CACHE[key] = {"ts": now, "report": report}
        return report
    except Exception as e:
        logger.error(f"wallet report error for {token_addr}: {e}")
        return {
            "risk_assessment": {"overall_risk": 100, "red_flags": ["error"]},
            "marketing_analysis": {"activity_score": 0, "total_spend_eth": 0},
            "developer_analysis": {"holding_percentage": 0},
            "wallet_summary": {},
        }



# ---------------------------------------------------------
# Wallet monitoring helpers (start/stop/list + resolver)
# ---------------------------------------------------------

def _resolve_main_token_from_arg(arg: str) -> Optional[str]:
    """Accepts a token or pair address; returns main token address or None."""
    if not arg:
        return None
    a = arg.strip()
    a_low = a.lower()
    # Direct token address (heuristic)
    if a_low.startswith("0x") and len(a_low) == 42:
        # If it's a known pair, convert to token
        if a_low in known_pairs:
            t0, t1 = known_pairs[a_low]
            try:
                return get_non_weth_token(t0, t1)
            except Exception:
                return t1 if t0.lower() == WETH_ADDRESS.lower() else t0
        # Otherwise assume token
        return a
    # Try to match known pairs by prefix
    for p, (t0, t1) in list(known_pairs.items()):
        if p.lower().startswith(a_low) or p.lower().endswith(a_low):
            try:
                return get_non_weth_token(t0, t1)
            except Exception:
                return t1 if t0.lower() == WETH_ADDRESS.lower() else t0
    return None


def start_wallet_monitor(token_addr: str):
    """Start background wallet monitoring if under limit."""
    key = token_addr.lower()
    if key in wallet_monitor_tasks or len(wallet_monitor_tasks) >= MAX_WALLET_MONITORS:
        return
    stop_event = asyncio.Event()
    wallet_monitor_stops[key] = stop_event
    try:
        fut = asyncio.run_coroutine_threadsafe(
            wallet_tracker.monitor_wallet_realtime(token_addr, wallet_activity_callback, stop_event),
            wallet_event_loop,
        )
        wallet_monitor_tasks[key] = fut
    except Exception as e:
        logger.error(f"wallet monitor error for {token_addr}: {e}")


def stop_wallet_monitor(token_or_pair_addr: str) -> bool:
    """Stop a running monitor. Returns True if stopped."""
    token = _resolve_main_token_from_arg(token_or_pair_addr) or token_or_pair_addr
    key = token.lower()
    fut = wallet_monitor_tasks.pop(key, None)
    ev = wallet_monitor_stops.pop(key, None)
    if ev is not None:
        try:
            ev.set()
        except Exception:
            pass
    if fut is not None:
        try:
            fut.cancel()
        except Exception:
            pass
        return True
    return False


def stop_all_wallet_monitors() -> int:
    keys = list(wallet_monitor_tasks.keys())
    count = 0
    for k in keys:
        try:
            if stop_wallet_monitor(k):
                count += 1
        except Exception:
            continue
    return count


def list_wallet_monitors() -> List[str]:
    """Return list of tokens currently being monitored."""
    return list(wallet_monitor_tasks.keys())


def detect_bull_market_early_gems(token_addr: str, pair_addr: str, wallet_report: dict = None) -> dict:
    """Detect projects with high potential in first 5-30 minutes"""
    indicators = {
        "rapid_buyer_growth": False,
        "no_early_dumps": False,
        "organic_growth": False,
        "dev_holding_stable": False,
        "smart_money_aping": False,
        "viral_potential": False,
        "score": 0,
    }

    if MARKET_MODE != "bull":
        return indicators

    ds = fetch_dexscreener_data(token_addr, pair_addr)
    metrics = analyze_transfer_history(token_addr, limit=50)

    mc = (ds.get("fdv") or ds.get("marketCap") or 0) if ds else 0
    liq = ds.get("liquidityUsd", 0) if ds else 0
    buyers = metrics.get("uniqueBuyers", 0)
    if (
        mc < MIN_GEM_MARKETCAP_USD
        or mc > MAX_GEM_MARKETCAP_USD
        or liq < MIN_GEM_LIQUIDITY_USD
        or buyers < MIN_GEM_UNIQUE_BUYERS
    ):
        return indicators

    if wallet_report:
        risk = wallet_report.get("risk_assessment", {}).get("overall_risk", 100)
        marketing = wallet_report.get("marketing_analysis", {}).get("activity_score", 0)
        if risk > MAX_GEM_RISK_SCORE or marketing < MIN_GEM_MARKETING_SCORE:
            return indicators

    if buyers >= MIN_GEM_UNIQUE_BUYERS:
        indicators["rapid_buyer_growth"] = True
        indicators["score"] += 25

    sells = ds.get("sells", 0) if ds else 0
    if liq > 0 and sells <= 3:
        indicators["no_early_dumps"] = True
        indicators["score"] += 20

    total_buys = ds.get("buys", 0) if ds else 0
    if total_buys > 0 and buyers / total_buys >= 0.5:
        indicators["organic_growth"] = True
        indicators["score"] += 15

    adv = advanced_contract_check(token_addr)
    if not adv.get("ownerActivity"):
        indicators["dev_holding_stable"] = True
        indicators["score"] += 10

    if metrics.get("smartMoneyCount", 0) > 0:
        indicators["smart_money_aping"] = True
        indicators["score"] += 20

    if ds and ds.get("baseTokenName") and re.search(
        r"(pepe|doge|floki|inu|meme|pump)", ds.get("baseTokenName"), re.IGNORECASE
    ):
        indicators["viral_potential"] = True
        indicators["score"] += 10

    if wallet_report:
        indicators["score"] += wallet_report.get("marketing_analysis", {}).get("activity_score", 0) * 0.2
        risk = wallet_report.get("risk_assessment", {}).get("overall_risk", 100)
        indicators["score"] += max(0, 50 - risk) * 0.2

    return indicators



def identify_bull_pump_patterns(pair_data: dict) -> dict:
    patterns = {
        "stealth_launch": False,
        "community_takeover": False,
        "meme_velocity": False,
        "whale_accumulation": False,
        "exchange_potential": False,
        "whitelist_blacklist": False,
        "dex_paid": False,
        "score": 0,
    }

    if not has_private_sale(pair_data.get("token")):
        patterns["stealth_launch"] = True
        patterns["score"] += 20

    if (
        pair_data.get("renounced")
        and pair_data.get("telegram_members_growth_rate", 0) > 50
    ):
        patterns["community_takeover"] = True
        patterns["score"] += 20

    name = str(pair_data.get("tokenName", ""))
    if re.search(r"(pepe|doge|floki|inu|meme|pump)", name, re.IGNORECASE):
        patterns["meme_velocity"] = True
        patterns["score"] += 10

    metrics = pair_data.get("onChainMetrics", {})
    hc = metrics.get("holderConcentration")
    sm = metrics.get("smartMoneyCount", 0)
    if hc is not None and hc < 0.1 and sm > 0:
        patterns["whale_accumulation"] = True
        patterns["score"] += 20

    if str(pair_data.get("contractCheckStatus", "")).lower() in {"ok", "verified"} and pair_data.get("riskScore", 0) < 5:
        patterns["exchange_potential"] = True
        patterns["score"] += 20

    risk_flags = pair_data.get("riskFlags", {}) or {}
    if risk_flags.get("canBlacklist") or risk_flags.get("botWhitelist"):
        patterns["whitelist_blacklist"] = True
        patterns["score"] += 60
        if risk_flags.get("canBlacklist") and risk_flags.get("botWhitelist"):
            patterns["score"] += 10

    if pair_data.get("dexPaid"):
        patterns["dex_paid"] = True
        patterns["score"] += 15

    return patterns


###########################################################
# Enhanced Bull Market Detection System
###########################################################


class BullMarketDetector:
    """Utility class for Graph based analytics"""

    def __init__(self):
        self.successful_patterns = self.load_historical_patterns()
        self.market_sentiment = self.calculate_market_sentiment()

    def query_graph(self, query: str) -> dict:
        """Execute GraphQL query against The Graph"""
        try:
            resp = requests.post(
                GRAPH_URL,
                json={"query": query},
                headers={"Authorization": f"Bearer {GRAPH_BEARER}"},
                timeout=10,
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {}

    def load_historical_patterns(self) -> dict:
        """Load successful token patterns from previous bull runs"""
        query = """
        {
            tokens(first:100, where:{ derivedETH_gt: "0.01" }, orderBy: tradeVolumeUSD, orderDirection: desc){
                id
                symbol
                name
                derivedETH
                tradeVolumeUSD
                totalLiquidity
                txCount
            }
        }
        """
        data = self.query_graph(query)
        patterns = {
            "volume_acceleration": [],
            "liquidity_growth": [],
            "holder_patterns": [],
            "price_trajectories": [],
        }
        for tok in data.get("data", {}).get("tokens", []):
            patterns["volume_acceleration"].append(self.analyze_volume_pattern(tok["id"]))
        return patterns

    def calculate_market_sentiment(self) -> float:
        """Calculate overall crypto market sentiment"""
        query = """
        {
            bundle(id: "1") { ethPrice }
            tokens(first:10, orderBy: tradeVolumeUSD, orderDirection: desc){ priceUSD volumeUSD totalValueLockedUSD }
        }
        """
        _data = self.query_graph(query)
        # Placeholder for real calculation
        return 75.0

    def analyze_volume_pattern(self, token_addr: str) -> dict:
        query = f"""
        {{
            tokenDayDatas(first:30, orderBy: date, orderDirection: desc, where:{{ token: "{token_addr.lower()}" }}){{
                date
                volumeUSD
            }}
        }}
        """
        data = self.query_graph(query)
        daily = data.get("data", {}).get("tokenDayDatas", [])
        if len(daily) < 2:
            return {"acceleration": 0, "consistency": 0, "trend": "flat"}
        vols = [float(d.get("volumeUSD", 0)) for d in daily]
        acceleration = float(np.gradient(vols).mean()) if vols else 0
        consistency = 1 / (1 + np.std(vols) / (np.mean(vols) + 1))
        return {
            "acceleration": acceleration,
            "consistency": consistency,
            "trend": "increasing" if acceleration > 0 else "decreasing",
        }

    def get_token_metrics(self, token_addr: str) -> dict:
        query = f"""
        {{
            token(id: "{token_addr.lower()}"){{
                tradeVolumeUSD
                totalLiquidity
                txCount
            }}
        }}
        """
        data = self.query_graph(query)
        t = data.get("data", {}).get("token", {}) or {}
        return {
            "volume": float(t.get("tradeVolumeUSD", 0)),
            "liquidity": float(t.get("totalLiquidity", 0)),
            "txCount": int(t.get("txCount", 0)),
        }

    def get_similar_tokens(self, token_addr: str, limit: int = 5) -> List[dict]:
        token_info = self.get_token_metrics(token_addr)
        query = (
            "{ tokens(first:50, where:{ totalLiquidity_gte: \"%f\", totalLiquidity_lte: \"%f\", txCount_gte: %d }, orderBy: tradeVolumeUSD, orderDirection: desc){ id symbol priceUSD tradeVolumeUSD totalLiquidity }}"
            % (
                token_info["liquidity"] * 0.5,
                token_info["liquidity"] * 2.0,
                max(token_info["txCount"] // 2, 1),
            )
        )
        data = self.query_graph(query)
        return data.get("data", {}).get("tokens", [])[:limit]

    def calculate_network_effect(self, token_addr: str) -> float:
        query = f"""
        {{
            pairs(where: {{ token0: "{token_addr.lower()}" }}){{ id volumeUSD reserveUSD }}
            pairs1: pairs(where: {{ token1: "{token_addr.lower()}" }}){{ id volumeUSD reserveUSD }}
        }}
        """
        data = self.query_graph(query)
        pairs = data.get("data", {}).get("pairs", [])
        pairs.extend(data.get("data", {}).get("pairs1", []))
        score = 0.0
        for p in pairs:
            vol = float(p.get("volumeUSD", 0))
            reserve = float(p.get("reserveUSD", 0))
            score += np.log1p(vol) * 0.3 + np.log1p(reserve) * 0.7
        return min(score / 10, 100.0)


class EnhancedBullGemDetector:
    def __init__(self, graph_detector: BullMarketDetector):
        self.graph = graph_detector
        self.weights = {
            "volume_velocity": 0.15,
            "holder_quality": 0.15,
            "network_effect": 0.10,
            "market_timing": 0.10,
            "pattern_match": 0.15,
            "social_momentum": 0.10,
            "liquidity_stability": 0.10,
            "smart_money": 0.15,
        }

    def detect_bull_market_early_gems(self, token_addr: str, pair_addr: str, onchain_data: dict | None) -> dict:
        onchain_data = onchain_data or {}
        scores: Dict[str, float] = {}
        vol_pat = self.graph.analyze_volume_pattern(token_addr)
        scores["volume_velocity"] = self.calculate_volume_velocity_score(vol_pat, onchain_data)
        scores["holder_quality"] = self.calculate_holder_quality_score(token_addr, onchain_data)
        scores["network_effect"] = self.graph.calculate_network_effect(token_addr)
        scores["market_timing"] = self.calculate_market_timing_score()
        scores["pattern_match"] = self.calculate_pattern_match_score(token_addr, self.graph.get_similar_tokens(token_addr))
        scores["social_momentum"] = self.calculate_social_momentum_score(onchain_data)
        scores["liquidity_stability"] = self.calculate_liquidity_stability_score(pair_addr)
        scores["smart_money"] = self.calculate_enhanced_smart_money_score(token_addr, onchain_data)

        total = sum(scores[k] * self.weights[k] for k in scores)
        gem_indicators = {
            "is_early_gem": total >= 70,
            "gem_confidence": min(total / 100, 1.0),
            "score": total,
            "breakdown": scores,
            "top_factors": self.get_top_factors(scores),
            "risk_factors": self.identify_risk_factors(scores),
            "recommended_action": self.get_recommendation(total, scores),
        }
        return gem_indicators

    def calculate_volume_velocity_score(self, volume_pattern: dict, onchain_data: dict | None) -> float:
        onchain_data = onchain_data or {}
        base = 0.0
        if volume_pattern.get("acceleration", 0) > 1000:
            base += 40
        elif volume_pattern.get("acceleration", 0) > 100:
            base += 25
        elif volume_pattern.get("acceleration", 0) > 0:
            base += 10
        base += volume_pattern.get("consistency", 0) * 30
        curr = onchain_data.get("volume24h", 0)
        if curr > 100000:
            base += 20
        elif curr > 25000:
            base += 10
        if onchain_data.get("age_minutes", 60) < 60 and onchain_data.get("buys", 0) > 50:
            base += 10
        return min(base, 100.0)

    def calculate_holder_quality_score(self, token_addr: str, onchain_data: dict | None) -> float:
        onchain_data = onchain_data or {}
        score = 0.0
        query = f"""
        {{
            token(id: "{token_addr.lower()}"){{ holders: holderCount }}
            transfers(first:100, where:{{ token:"{token_addr.lower()}" }}, orderBy: timestamp, orderDirection: desc){{ from to amount }}
        }}
        """
        data = self.graph.query_graph(query)
        holders = data.get("data", {}).get("token", {}).get("holders", 0)
        if holders > 100:
            score += 30
        elif holders > 50:
            score += 20
        elif holders > 20:
            score += 10
        transfers = data.get("data", {}).get("transfers", [])
        unique_buyers = len({t.get("to") for t in transfers})
        unique_sellers = len({t.get("from") for t in transfers})
        if unique_buyers > unique_sellers * 2:
            score += 30
        elif unique_buyers > unique_sellers:
            score += 15
        if onchain_data.get("holderConcentration", 1.0) < 0.3:
            score += 20
        elif onchain_data.get("holderConcentration", 1.0) < 0.5:
            score += 10
        if self.detect_organic_pattern(transfers):
            score += 20
        return min(score, 100.0)

    def calculate_market_timing_score(self) -> float:
        sentiment = self.graph.market_sentiment
        if 40 <= sentiment <= 70:
            return 80.0
        if 30 <= sentiment <= 80:
            return 60.0
        if sentiment > 80:
            return 40.0
        return 20.0

    def calculate_pattern_match_score(self, token_addr: str, similar_tokens: List[dict]) -> float:
        if not similar_tokens:
            return 50.0
        score = 0.0
        current = self.graph.get_token_metrics(token_addr)
        for s in similar_tokens:
            if current.get("volume", 0) > float(s.get("tradeVolumeUSD", 0)) * 0.1:
                score += 20
        return min(score, 100.0)

    def calculate_social_momentum_score(self, onchain_data: dict | None) -> float:
        onchain_data = onchain_data or {}
        score = 0.0
        name = str(onchain_data.get("tokenName", "")).lower()
        viral_terms = [
            "pepe",
            "doge",
            "shib",
            "floki",
            "inu",
            "moon",
            "rocket",
            "elon",
            "trump",
            "lambo",
            "wagmi",
            "gm",
            "based",
            "chad",
        ]
        viral_count = sum(1 for term in viral_terms if term in name)
        score += min(viral_count * 15, 30)
        links = onchain_data.get("socialLinks", [])
        if any("twitter" in l for l in links):
            score += 20
        if any("telegram" in l for l in links):
            score += 20
        if any("discord" in l for l in links):
            score += 10
        buys = onchain_data.get("buys", 0)
        age_hours = onchain_data.get("age_minutes", 60) / 60
        if age_hours > 0:
            bph = buys / age_hours
            if bph > 100:
                score += 20
            elif bph > 50:
                score += 10
        return min(score, 100.0)

    def calculate_liquidity_stability_score(self, pair_addr: str) -> float:
        query = f"""
        {{
            pairDayDatas(first:7, orderBy: date, orderDirection: desc, where:{{ pairAddress: "{pair_addr.lower()}" }}){{ date reserveUSD }}
        }}
        """
        data = self.graph.query_graph(query)
        daily = data.get("data", {}).get("pairDayDatas", [])
        if not daily:
            return 50.0
        reserves = [float(d.get("reserveUSD", 0)) for d in daily]
        if len(reserves) >= 2:
            if reserves[0] > reserves[-1] * 1.5:
                return 90.0
            if reserves[0] > reserves[-1] * 1.2:
                return 70.0
            if reserves[0] > reserves[-1]:
                return 60.0
        return 40.0

    def calculate_enhanced_smart_money_score(self, token_addr: str, onchain_data: dict | None) -> float:
        onchain_data = onchain_data or {}
        score = onchain_data.get("smartMoneyCount", 0) * 10
        query = f"""
        {{
            transfers(first:50, where:{{ token:"{token_addr.lower()}", amount_gt: "1000000" }}, orderBy: amount, orderDirection: desc){{ from to amount }}
        }}
        """
        data = self.graph.query_graph(query)
        transfers = data.get("data", {}).get("transfers", [])
        whale_buys = sum(1 for t in transfers if t.get("to") not in BASE_TOKENS)
        whale_sells = sum(1 for t in transfers if t.get("from") not in BASE_TOKENS)
        if whale_buys > whale_sells * 2:
            score += 40
        elif whale_buys > whale_sells:
            score += 20
        return min(score, 100.0)

    def detect_organic_pattern(self, transfers: List[dict]) -> bool:
        if len(transfers) < 10:
            return True
        amounts = [float(t.get("amount", 0)) for t in transfers[:20]]
        if amounts:
            std = np.std(amounts)
            mean = np.mean(amounts)
            if mean > 0 and std / mean < 0.1:
                return False
        return True

    def get_top_factors(self, scores: dict) -> List[str]:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1] * self.weights.get(x[0], 0), reverse=True)
        return [k for k, _ in sorted_scores[:3] if scores.get(k, 0) > 50]

    def identify_risk_factors(self, scores: dict) -> List[str]:
        risks = []
        if scores.get("liquidity_stability", 0) < 40:
            risks.append("Unstable liquidity")
        if scores.get("holder_quality", 0) < 30:
            risks.append("Poor holder distribution")
        if scores.get("market_timing", 0) < 30:
            risks.append("Unfavorable market conditions")
        if scores.get("pattern_match", 0) < 20:
            risks.append("Doesn't match successful patterns")
        return risks

    def get_recommendation(self, total: float, scores: dict) -> str:
        if total >= 80:
            return "🚀 HIGH CONVICTION - Strong buy signal"
        if total >= 70:
            return "✅ GOOD OPPORTUNITY - Consider entry"
        if total >= 60:
            return "👀 WATCH CLOSELY - Wait for confirmation"
        if total >= 50:
            # Avoid potential unicode parsing issues on some platforms
            return "RISKY - Only for high risk tolerance (\u26A0)"
        return "❌ AVOID - Too many red flags"


class EnhancedBullPatternAnalyzer:
    def __init__(self, graph_detector: BullMarketDetector):
        self.graph = graph_detector
        self.patterns = {
            "stealth_ninja": self.detect_stealth_ninja_launch,
            "community_fomo": self.detect_community_fomo_pattern,
            "whale_accumulation": self.detect_whale_accumulation,
            "exchange_rush": self.detect_exchange_rush_pattern,
            "meme_supercycle": self.detect_meme_supercycle,
            "defi_rotation": self.detect_defi_rotation,
            "copy_trade_magnet": self.detect_copy_trade_pattern,
        }

    def identify_bull_pump_patterns(self, token_data: dict) -> dict:
        detected = {}
        confidences = []
        for name, func in self.patterns.items():
            res = func(token_data)
            if res["detected"]:
                detected[name] = res
                confidences.append(res["confidence"])
        avg_score = sum(confidences) / len(confidences) if confidences else 0.0
        dominant = max(detected, key=lambda k: detected[k]["confidence"]) if detected else None
        return {
            "patterns_detected": list(detected.keys()),
            "pattern_details": detected,
            "score": min(avg_score, 100.0),
            "dominant_pattern": dominant,
            "pump_probability": self.calculate_pump_probability(detected),
            "recommended_strategy": self.get_strategy_recommendation(detected),
        }

    def detect_stealth_ninja_launch(self, token_data: dict) -> dict:
        indicators = {
            "no_presale": not token_data.get("privateSale", {}).get("hasPresale", False),
            "low_initial_liquidity": token_data.get("liquidityUsd", 0) < 50000,
            "rapid_growth": token_data.get("buys", 0) > 30 and in_first_hour(token_data),
            "fair_launch": token_data.get("contractRenounced", False),
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 60, "confidence": conf, "indicators": indicators, "description": "Stealth launch with organic growth potential"}

    def detect_community_fomo_pattern(self, token_data: dict) -> dict:
        growth = self.analyze_social_growth(token_data)
        indicators = {
            "telegram_explosive": growth.get("telegram_growth_rate", 0) > 100,
            "twitter_viral": growth.get("twitter_mentions", 0) > 50,
            "holder_rush": token_data.get("uniqueBuyers", 0) > 100,
            "no_major_sells": token_data.get("sells", 0) < token_data.get("buys", 0) * 0.1,
            "increasing_volume": self.check_volume_trend(token_data),
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 70, "confidence": conf, "indicators": indicators, "description": "Community FOMO building rapidly"}

    def detect_whale_accumulation(self, token_data: dict) -> dict:
        query = f"""
        {{
            transfers(
                where: {{ token: "{token_data['token'].lower()}", amount_gt: "10000" }},
                orderBy: timestamp,
                orderDirection: asc
            ) {{
                from
                to
                amount
                timestamp
            }}
        }}
        """
        data = self.graph.query_graph(query)
        transfers = data.get("data", {}).get("transfers", [])
        whale_wallets = set()
        score = 0
        for t in transfers:
            if t.get("to") not in BASE_TOKENS:
                whale_wallets.add(t.get("to"))
                score += 1
            if t.get("from") in whale_wallets:
                score -= 0.5
        indicators = {
            "whale_count": len(whale_wallets) > 3,
            "accumulation_trend": score > 5,
            "smart_money_present": token_data.get("smartMoneyCount", 0) > 0,
            "stable_liquidity": token_data.get("liquidityUsd", 0) > 100000,
            "low_sell_pressure": token_data.get("clogPercent", 100) < 30,
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 60, "confidence": conf, "indicators": indicators, "whale_addresses": list(whale_wallets)[:5], "description": f"{len(whale_wallets)} whales accumulating"}

    def detect_exchange_rush_pattern(self, token_data: dict) -> dict:
        indicators = {
            "high_volume": token_data.get("volume24h", 0) > 500000,
            "many_holders": self.get_holder_count(token_data["token"]) > 500,
            "verified_contract": token_data.get("contractCheckStatus") == "OK",
            "low_risk": token_data.get("riskScore", 100) < 5,
            "professional_setup": bool(token_data.get("socialLinks", [])),
            "locked_liquidity": token_data.get("lockedLiquidity", False),
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 70, "confidence": conf, "indicators": indicators, "description": "Professional setup indicating exchange ambitions"}

    def detect_meme_supercycle(self, token_data: dict) -> dict:
        name = token_data.get("tokenName", "").lower()
        meme_keywords = {
            "tier1": ["pepe", "doge", "shib", "floki"],
            "tier2": ["inu", "elon", "moon", "chad", "wojak"],
            "tier3": ["gm", "wagmi", "based", "frog", "cat", "pnut"],
        }
        meme_score = 0
        for tier, keywords in meme_keywords.items():
            for kw in keywords:
                if kw in name:
                    meme_score += {"tier1": 30, "tier2": 20, "tier3": 10}[tier]
        meme_market_hot = self.check_meme_market_temperature()
        indicators = {
            "meme_name": meme_score > 0,
            "viral_potential": meme_score >= 30,
            "meme_market_hot": meme_market_hot > 70,
            "community_driven": token_data.get("contractRenounced", False),
            "high_tx_count": token_data.get("buys", 0) + token_data.get("sells", 0) > 100,
            "social_presence": len(token_data.get("socialLinks", [])) > 0,
        }
        conf = (sum(indicators.values()) / len(indicators)) * 100
        conf *= (1 + meme_score / 100)
        return {"detected": conf > 60, "confidence": min(conf, 100.0), "indicators": indicators, "meme_score": meme_score, "description": f"Meme potential in {meme_market_hot}% hot market"}

    def detect_defi_rotation(self, token_data: dict) -> dict:
        defi = self.check_defi_characteristics(token_data)
        indicators = {
            "defi_features": defi.get("has_defi_features"),
            "high_apy_potential": defi.get("yield_potential", 0) > 100,
            "tvl_growing": self.check_tvl_growth(token_data["token"]),
            "institutional_interest": token_data.get("liquidityUsd", 0) > 500000,
            "low_risk_profile": token_data.get("riskScore", 100) < 10,
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 60, "confidence": conf, "indicators": indicators, "description": "DeFi rotation opportunity detected"}

    def detect_copy_trade_pattern(self, token_data: dict) -> dict:
        notable_buyers = self.identify_notable_buyers(token_data["token"])
        indicators = {
            "followed_wallets": len(notable_buyers) > 0,
            "consistent_buys": self.check_consistent_buying(token_data["token"]),
            "no_insider_dumps": not self.detect_insider_selling(token_data),
            "growing_interest": token_data.get("buys", 0) > token_data.get("sells", 0) * 3,
            "alpha_seekers": token_data.get("smartMoneyCount", 0) > 2,
        }
        conf = sum(indicators.values()) / len(indicators) * 100
        return {"detected": conf > 60, "confidence": conf, "indicators": indicators, "notable_buyers": notable_buyers[:3], "description": f"{len(notable_buyers)} notable wallets accumulating"}

    def calculate_pump_probability(self, patterns: dict) -> float:
        if not patterns:
            return 10.0
        weights = {
            "whale_accumulation": 0.25,
            "community_fomo": 0.20,
            "meme_supercycle": 0.15,
            "exchange_rush": 0.15,
            "stealth_ninja": 0.10,
            "copy_trade_magnet": 0.10,
            "defi_rotation": 0.05,
        }
        prob = 0.0
        for p, d in patterns.items():
            prob += d["confidence"] * weights.get(p, 0.1)
        prob *= (0.5 + 0.5 * self.graph.market_sentiment / 100)
        return min(prob, 95.0)

    def get_strategy_recommendation(self, patterns: dict) -> dict:
        if not patterns:
            return {"action": "SKIP", "reasoning": "No significant patterns detected", "risk_level": "N/A"}
        dom = max(patterns, key=lambda k: patterns[k]["confidence"])
        strategies = {
            "whale_accumulation": {"action": "ACCUMULATE", "reasoning": "Whales accumulating", "risk_level": "LOW", "position_size": "5%"},
            "community_fomo": {"action": "BUY BREAKOUT", "reasoning": "Community FOMO", "risk_level": "MEDIUM", "position_size": "3%"},
            "exchange_rush": {"action": "Accumulate before listing", "exit": "Sell on exchange announcement", "risk_level": "MEDIUM", "position_size": "2-4% of portfolio"},
        }
        return strategies.get(dom, {"action": "RESEARCH MORE", "reasoning": f"{dom} pattern needs investigation", "risk_level": "UNKNOWN"})

    # ----- helper stubs -----
    def check_volume_trend(self, token_data: dict) -> bool:
        vp = self.graph.analyze_volume_pattern(token_data["token"])
        return vp.get("trend") == "increasing"

    def get_holder_count(self, token_addr: str) -> int:
        query = f"{{ token(id: \"{token_addr.lower()}\"){{ holderCount }} }}"
        data = self.graph.query_graph(query)
        return data.get("data", {}).get("token", {}).get("holderCount", 0)

    def check_meme_market_temperature(self) -> float:
        return 0.0

    def analyze_social_growth(self, token_data: dict) -> dict:
        return {"telegram_growth_rate": 0, "twitter_mentions": 0}

    def check_defi_characteristics(self, token_data: dict) -> dict:
        return {"has_defi_features": False, "yield_potential": 0}

    def check_tvl_growth(self, token_addr: str) -> bool:
        return False

    def identify_notable_buyers(self, token_addr: str) -> List[str]:
        return []

    def check_consistent_buying(self, token_addr: str) -> bool:
        return False

    def detect_insider_selling(self, token_data: dict) -> bool:
        return False


def in_first_hour(token_data: dict) -> bool:
    return token_data.get("age_minutes", 61) <= 60


def integrate_enhanced_detection(token_addr: str, pair_addr: str, existing_data: dict) -> dict:
    graph_detector = BullMarketDetector()
    gem_detector = EnhancedBullGemDetector(graph_detector)
    pattern_analyzer = EnhancedBullPatternAnalyzer(graph_detector)
    gem_results = gem_detector.detect_bull_market_early_gems(token_addr, pair_addr, existing_data)
    pattern_results = pattern_analyzer.identify_bull_pump_patterns({**existing_data, "token": token_addr})
    return {
        "gem_analysis": gem_results,
        "pattern_analysis": pattern_results,
        "combined_score": (gem_results["score"] + pattern_results["score"]) / 2,
        "action_required": gem_results["score"] >= 70 or pattern_results["score"] >= 70,
        "priority_level": calculate_priority(gem_results, pattern_results),
    }


def calculate_priority(gem_results: dict, pattern_results: dict) -> str:
    combined = (gem_results["score"] + pattern_results["score"]) / 2
    if combined >= 85:
        return "🔴 CRITICAL - Immediate action"
    if combined >= 70:
        return "🟠 HIGH - Strong opportunity"
    if combined >= 55:
        return "🟡 MEDIUM - Worth watching"
    return "🟢 LOW - Monitor only"


def send_enhanced_alert(pair_addr: str, results: dict):
    msg = f"[EnhancedBull] <code>{pair_addr}</code>\n"
    msg += f"Gem Score: {results['gem_analysis']['score']:.1f}\n"
    msg += f"Pattern Score: {results['pattern_analysis']['score']:.1f}\n"
    msg += f"Priority: {results['priority_level']}"
    send_telegram_message(msg)


def send_borderline_warning(
    pair_addr: str,
    mc: float,
    liq: float,
    risk_score: float,
    marketing_score: float,
    wl_bl: bool,
):
    """Alert when a project narrowly misses gem criteria."""
    msg = f"[GemWarning] <code>{pair_addr}</code>\n"
    msg += f"MC: ${mc:,.0f} | Liq: ${liq:,.0f}\n"
    msg += f"Risk: {risk_score:.1f} | Marketing: {marketing_score:.1f}"
    if wl_bl:
        msg += "\nWhitelist/Blacklist present"
    msg += "\nBorderline metrics - monitor closely"
    send_telegram_message(msg)


###########################################################
# 9. FIRST SELL DETECTION HELPERS
###########################################################


def _address_in_source(token_addr: str, wallet: str) -> bool:
    info = fetch_contract_source_etherscan(token_addr)
    if info.get("status") != ContractVerificationStatus.VERIFIED:
        return False
    w = wallet.lower().replace("0x", "")
    for part in info.get("source", []):
        if w in part.get("content", "").lower():
            return True
    return False


def analyze_seller_wallet(token_addr: str, wallet: str) -> Tuple[str, int]:
    """Return descriptive flags and a simple risk score for a seller wallet."""
    flags = []
    risk = 0
    try:
        code = w3_read.eth.get_code(to_checksum_address(wallet))
        if code and len(code) > 0:
            flags.append("contract")
            risk += 5
    except Exception:
        pass
    try:
        if _address_in_source(token_addr, wallet):
            flags.append("listed_in_contract")
            risk += 2
    except Exception:
        pass
    try:
        txc = w3_read.eth.get_transaction_count(to_checksum_address(wallet))
        if txc < 5:
            flags.append("new_wallet")
            risk += 1
    except Exception:
        pass
    try:
        bal = w3_read.eth.get_balance(to_checksum_address(wallet))
        if bal < Web3.to_wei(0.05, "ether"):
            flags.append("low_balance")
            risk += 1
    except Exception:
        pass
    if not flags:
        flags.append("EOA")
    return ", ".join(flags), risk


def detect_first_sell(pair_addr: str, token0: str, token1: str, from_block: int) -> Optional[str]:
    try:
        logs = w3_read.eth.get_logs(
            {
                "address": pair_addr,
                "fromBlock": from_block,
                "toBlock": "latest",
                "topics": [SWAP_TOPIC_V2],
            }
        )
        for lg in logs:
            data_field = lg["data"]
            if isinstance(data_field, HexBytes):
                data_field = data_field.hex()
            if data_field.startswith("0x"):
                data_field = data_field[2:]
            a0_in, a1_in, a0_out, a1_out = decode(
                ["uint256", "uint256", "uint256", "uint256"],
                bytes.fromhex(data_field),
            )
            is_sell = False
            if token0.lower() == WETH_ADDRESS.lower():
                is_sell = a0_out > 0
            elif token1.lower() == WETH_ADDRESS.lower():
                is_sell = a1_out > 0
            if is_sell:
                try:
                    tx = w3_read.eth.get_transaction(lg["transactionHash"])
                    return tx["from"]
                except Exception:
                    sender = "0x" + lg["topics"][1].hex()[-40:]
                    return sender
    except Exception as e:
        logger.debug(f"first sell v2 error: {e}")

    try:
        logs = w3_read.eth.get_logs(
            {
                "address": pair_addr,
                "fromBlock": from_block,
                "toBlock": "latest",
                "topics": [SWAP_TOPIC_V3],
            }
        )
        for lg in logs:
            data_field = lg["data"]
            if isinstance(data_field, HexBytes):
                data_field = data_field.hex()
            if data_field.startswith("0x"):
                data_field = data_field[2:]
            a0, a1, _, _, _ = decode(
                ["int256", "int256", "uint160", "uint128", "int24"],
                bytes.fromhex(data_field),
            )
            is_sell = False
            if token0.lower() == WETH_ADDRESS.lower():
                is_sell = a0 < 0
            elif token1.lower() == WETH_ADDRESS.lower():
                is_sell = a1 < 0
            if is_sell:
                try:
                    tx = w3_read.eth.get_transaction(lg["transactionHash"])
                    return tx["from"]
                except Exception:
                    sender = "0x" + lg["topics"][1].hex()[-40:]
                    return sender
    except Exception as e:
        logger.debug(f"first sell v3 error: {e}")
    return None


###########################################################
# 9. MAIN PAIR CRITERIA (extended with advanced checks)
###########################################################

detected_at: Dict[str, float] = {}


def check_pair_criteria(
    pair_addr: str, token0: str, token1: str
) -> Tuple[int, int, dict]:
    # advanced contract check + renounce status + dex metrics + wallet analysis
    total_checks = 13
    passes = 0

    # determine main token (non-WETH)
    main_token = token1 if token0.lower() == WETH_ADDRESS.lower() else token0

    # 1) fetch DexScreener first to filter out unlisted or tiny pairs
    dex_data = fetch_dexscreener_data(main_token, pair_addr)
    if not dex_data:
        logger.warning(f"[DexMissing] {pair_addr} missing DexScreener data")
        return (0, total_checks, {"dexscreener_missing": True})
    if dex_data["liquidityUsd"] <= 1:
        return (0, total_checks, {"tokenName": "Rugpull"})
    liq = dex_data["liquidityUsd"]
    mc = dex_data["marketCap"]
    if liq < MIN_LIQUIDITY_USD or mc < MIN_MARKETCAP_USD:
        # Skip illiquid or microcap tokens to reduce noise
        return (0, total_checks, {})

    price = dex_data["priceUsd"]
    vol = dex_data["volume24h"]
    fdv = dex_data["fdv"]
    buys = dex_data["buys"]
    sells = dex_data["sells"]
    locked_liq = dex_data["lockedLiquidity"]

    # 2) advanced contract verification
    adv = advanced_contract_check(main_token)
    if adv.get("riskFlags", {}).get("walletDrainer"):
        return (0, total_checks, {"tokenName": "WalletDrainer"})
    if adv["verified"] and adv["riskScore"] < 10:
        passes += 1

    # 3) contract renounced requirement
    if adv.get("renounced"):
        passes += 1

    # 4) honeypot check (only for DexScreener-listed pairs)
    hp0 = check_honeypot_is(token0, pair_addr=pair_addr)
    hp1 = check_honeypot_is(token1, pair_addr=pair_addr)
    if hp0 or hp1:
        logger.info(f"[Honeypot DETECTED] => pair={pair_addr}")
        return (0, total_checks, {"tokenName": "Honeypot"})

    # 5) renounced info for both tokens (for logging only)
    ren0 = check_is_renounced(token0)
    ren1 = check_is_renounced(token1)
    if ren0 and ren1:
        passes += 1

    # 6) liq≥MIN_LIQUIDITY_USD
    passes += 1  # baseline already ensured, count as pass

    # 7) vol≥MIN_VOLUME_USD
    if vol >= MIN_VOLUME_USD:
        passes += 1

    # 8) FDV≥MIN_FDV_USD & MC≥MIN_MARKETCAP_USD
    if fdv >= MIN_FDV_USD and mc >= MIN_MARKETCAP_USD:
        passes += 1

    # 9) price>0
    if price > 0:
        passes += 1

    # 10) liquidity locked
    if locked_liq:
        passes += 1

    clog = 0.0
    if buys + sells > 0:
        clog = (sells / (buys + sells)) * 100

    # 11) buys≥MIN_BUYS_FIRST_HOUR
    pkey = pair_addr.lower()
    if pkey not in detected_at:
        detected_at[pkey] = time.time()
    elapsed = time.time() - detected_at[pkey]
    if elapsed < 3600:
        if buys >= MIN_BUYS_FIRST_HOUR:
            passes += 1
    else:
        passes += 1

    # 12 & 13) wallet analysis
    wallet_report = get_wallet_report(main_token)
    if wallet_report["risk_assessment"].get("overall_risk", 100) < 50:
        passes += 1
    if wallet_report["marketing_analysis"].get("activity_score", 0) > 30:
        passes += 1

    extra = {
        "tokenName": dex_data["baseTokenName"],
        "logoUrl": dex_data["baseTokenLogo"],
        "priceUsd": price,
        "liquidityUsd": liq,
        "volume24h": vol,
        "fdv": fdv,
        "marketCap": mc,
        "buys": buys,
        "sells": dex_data["sells"],
        "socialLinks": dex_data.get("socialLinks", []),
        "lockedLiquidity": dex_data["lockedLiquidity"],
        "clogPercent": clog,
        "renounced": (ren0 and ren1),
        # Also store advanced check
        "contractCheckStatus": adv["status"],
        "verified": adv.get("verified", False),
        "riskScore": adv["riskScore"],
        "owner": adv.get("owner"),
        "ownerBalanceEth": adv.get("ownerBalanceEth"),
        "ownerTokenBalance": adv.get("ownerTokenBalance"),
        "implementation": adv.get("implementation"),
        "contractRenounced": adv.get("renounced"),
        "slitherIssues": adv.get("slitherIssues"),
        "privateSale": adv.get("privateSale"),
        "onChainMetrics": adv.get("onChainMetrics"),
        "riskFlags": adv.get("riskFlags", {}),
        "dexPaid": dex_data.get("dexPaid"),
        "wallet_analysis": {
            "risk_score": wallet_report["risk_assessment"].get("overall_risk", 100),
            "marketing_score": wallet_report["marketing_analysis"].get("activity_score", 0),
            "marketing_spend_eth": wallet_report["marketing_analysis"].get("total_spend_eth", 0),
            "dev_holding_percentage": wallet_report["developer_analysis"].get("holding_percentage", 0),
            "red_flags": wallet_report["risk_assessment"].get("red_flags", []),
            "positive_signals": wallet_report["risk_assessment"].get("positive_signals", []),
        },
    }
    return (passes, total_checks, extra)


###########################################################
# 9.1 SEND UI MESSAGE
###########################################################


def send_ui_criteria_message(
    pair_addr: str,
    passes: int,
    total: int,
    is_recheck: bool = False,
    token_name: str = "",
    clog_percent: float = None,
    logo_url: str = None,
    extra_stats: Dict = None,
    recheck_attempt: int = None,
    is_passing_refresh: bool = False,
):
    """
    Only send if passes >= MIN_PASS_THRESHOLD
    """
    if passes < MIN_PASS_THRESHOLD:
        return
    if is_passing_refresh:
        prefix = "[Refresh]"
    else:
        prefix = "[Recheck]" if is_recheck else "[NewPair]"
    attempt_str = f" (Attempt #{recheck_attempt})" if recheck_attempt else ""
    pass_str = f"{passes}/{total} passes"
    tn = token_name or "Unnamed"

    msg = (
        f"🟢 <b>{tn}</b> {prefix}{attempt_str}\n"
        f"Pair: <code>{pair_addr}</code>\n"
        f"Criteria: <b>{pass_str}</b>"
    )
    if clog_percent is not None:
        msg += f"\nClog: {clog_percent:.0f}% sells"
    if extra_stats:
        pr = extra_stats.get("priceUsd")
        li = extra_stats.get("liquidityUsd")
        vo = extra_stats.get("volume24h")
        fdv = extra_stats.get("fdv")
        mc = extra_stats.get("marketCap")
        buys = extra_stats.get("buys")
        sells = extra_stats.get("sells")
        ren = extra_stats.get("renounced", False)
        locked = extra_stats.get("lockedLiquidity", False)
        owner_addr = extra_stats.get("owner")
        owner_bal = extra_stats.get("ownerBalanceEth")
        owner_tok = extra_stats.get("ownerTokenBalance")
        impl = extra_stats.get("implementation")
        contract_renounced = extra_stats.get("contractRenounced")
        slither_issues = extra_stats.get("slitherIssues")
        psale = extra_stats.get("privateSale", {})
        metrics = extra_stats.get("onChainMetrics", {})

        msg += "\n\n<b>Dex Stats:</b>"
        if pr and pr > 0:
            msg += f"\nPrice: ${pr:,.12f}"
        if li is not None:
            msg += f"\nLiquidity: ${li:,.0f}"
        if vo is not None:
            msg += f"\n24h Volume: ${vo:,.0f}"
        if fdv:
            msg += f"\nFDV: ${fdv:,.0f}"
        if mc:
            msg += f"\nM.Cap: ${mc:,.0f}"
        msg += f"\nBuys/Sells: {buys}/{sells}"
        if locked is not None:
            msg += f"\nLiquidity Locked: <b>{'Yes' if locked else 'No'}</b>"
        cstat = extra_stats.get("contractCheckStatus")
        rscore = extra_stats.get("riskScore")
        verified_bool = bool(extra_stats.get("verified") is True)
        if rscore is not None:
            msg += f"\nVerified: <b>{'Yes' if verified_bool else 'No'}</b> | Risk Score: {rscore}"
        links = extra_stats.get("socialLinks", [])
        if links:
            msg += "\n\n<b>Links:</b>"
            for lk in links:
                msg += f"\n{lk}"
        if owner_addr:
            msg += f"\nOwner: {owner_addr}"
        if owner_addr and owner_addr.lower() != ZERO_ADDRESS.lower() and owner_bal is not None:
            msg += f"\nOwner ETH: {owner_bal:.4f}"
        if owner_addr and owner_addr.lower() != ZERO_ADDRESS.lower() and owner_tok is not None:
            msg += f"\nOwner Token Bal: {owner_tok}"
        if impl:
            msg += f"\nImpl: {impl}"
        if contract_renounced is not None:
            msg += (
                f"\nContract Renounced: <b>{'Yes' if contract_renounced else 'No'}</b>"
            )
        if psale.get("hasPresale"):
            count = len(psale.get("largeTransfers", []))
            msg += f"\nPrivate Sale: {count} large transfers"
        if metrics:
            hc = metrics.get("holderConcentration")
            ratio = metrics.get("uniqueBuyerSellerRatio")
            sm = metrics.get("smartMoneyCount")
            if hc is not None:
                msg += f"\nHolder Concentration: {hc:.2%}"
            if ratio is not None:
                msg += f"\nBuyer/Seller Ratio: {ratio:.2f}"
            if sm:
                msg += f"\nSmart Money Buys: {sm}"
        if slither_issues not in (None, "error"):
            msg += f"\nSlither Issues: {slither_issues}"
        elif slither_issues == "error":
            msg += "\nSlither: error"

    send_telegram_message(msg)


def evaluate_fail_reasons(extra: Dict) -> List[str]:
    """Return list of failure reasons based on statistics."""
    reasons: List[str] = []
    mc = extra.get("marketCap", 0)
    liq = extra.get("liquidityUsd", 0)
    fdv = extra.get("fdv", 0)
    buys = extra.get("buys", 0)
    sells = extra.get("sells", 0)
    risk = extra.get("riskScore", 0)
    renounced_contract = extra.get("contractRenounced")
    slither_issues = extra.get("slitherIssues")

    if mc >= 100_000 and (buys + sells) < 10:
        reasons.append("High market cap with <10 buys/sells")
    if liq < MIN_LIQUIDITY_USD:
        reasons.append(f"Liquidity below ${MIN_LIQUIDITY_USD}")
    if fdv < MIN_FDV_USD or mc < MIN_MARKETCAP_USD:
        reasons.append("FDV/MC below thresholds")
    if risk >= 10:
        reasons.append("Contract high risk")
    if renounced_contract is False:
        reasons.append("Contract not renounced")
    if isinstance(slither_issues, int) and slither_issues >= 5:
        reasons.append(f"Slither issues {slither_issues}")

    return reasons


def send_bull_insights(pair_addr: str, token_name: str, gem: dict, patterns: dict):
    if not gem and not patterns:
        return
    if (
        gem.get("score", 0) < GEM_ALERT_SCORE
        and patterns.get("score", 0) < GEM_ALERT_SCORE
        and not patterns.get("whitelist_blacklist")
    ):
        return
    msg = f"[BullIndicators] <b>{token_name or 'Unnamed'}</b>\nPair: <code>{pair_addr}</code>"
    if gem.get("score", 0) > 0:
        msg += f"\nEarlyGem Score: {gem['score']}"
        parts = [k.replace('_', ' ').title() for k, v in gem.items() if k != 'score' and v]
        if parts:
            msg += "\n" + ", ".join(parts)
    if patterns.get("score", 0) > 0:
        msg += f"\nPump Pattern Score: {patterns['score']}"
        parts = [k.replace('_', ' ').title() for k, v in patterns.items() if k != 'score' and v]
        if parts:
            msg += "\n" + ", ".join(parts)
    if patterns.get("dex_paid"):
        msg += "\nDex Paid Promotion Detected"
    if patterns.get("whitelist_blacklist"):
        msg += "\nWhitelist/Blacklist Functions Present"
    logger.info(msg)
    send_telegram_message(msg)


def critical_verification_failure(extra: Dict) -> Tuple[bool, str]:
    """Return True if contract is unverified with error or risk score 9999."""
    status = str(extra.get("contractCheckStatus", "")).upper()
    risk = extra.get("riskScore")
    if status == "ERROR":
        return True, "verification error"
    if risk == 9999:
        return True, "risk score 9999"
    return False, ""


def store_pair_record(pair: str, token0: str, token1: str, passes: int, total: int, extra: Dict):
    if not isinstance(extra, dict):
        extra = {}
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active
    found = False
    for row in ws.iter_rows(min_row=2):
        if str(row[0].value).lower() == pair.lower():
            row[1].value = token0
            row[2].value = token1
            row[3].value = int(time.time())
            row[4].value = passes
            row[5].value = total
            row[6].value = extra.get("liquidityUsd", 0)
            row[7].value = extra.get("volume24h", 0)
            row[8].value = extra.get("marketCap", 0)
            found = True
            break
    if not found:
        ws.append([
            pair,
            token0,
            token1,
            int(time.time()),
            passes,
            total,
            extra.get("liquidityUsd", 0),
            extra.get("volume24h", 0),
            extra.get("marketCap", 0),
            "",
            "",
        ])
    wb.save(EXCEL_FILE)


def record_first_sell(pair: str, seller: str, block: int):
    wb = load_workbook(EXCEL_FILE)
    ws = wb.active
    for row in ws.iter_rows(min_row=2):
        if str(row[0].value).lower() == pair.lower():
            row[9].value = seller
            row[10].value = block
            break
    wb.save(EXCEL_FILE)


###########################################################
# 10. PASSING PAIRS
###########################################################

passing_pairs: Dict[str, dict] = {}

# Pairs waiting to hit minimum volume/trade requirements before promotion
volume_checks: Dict[str, dict] = {}


def queue_passing_refresh(
    pair_addr: str, token0: str, token1: str, init_mc: float, init_liq: float
):
    if pair_addr not in passing_pairs:
        passing_pairs[pair_addr] = {
            "token0": token0,
            "token1": token1,
            "attempt_index": 0,
            "last_attempt": time.time(),
            "initial_mc": init_mc,
            "mc_milestones_hit": set(),
            "silent_active": True,
            "last_silent_check": time.time(),
            "last_silent_mc": init_mc,
            "last_liq_check": time.time(),
            "last_liquidity": init_liq,
            "no_new_high_count": 0,
            # For quick 1-min honeypot check
            "last_hp_check": 0,
            # To avoid repeated spam logs
            "no_more_attempts_logged": False,
            "lp_supply": get_lp_total_supply(pair_addr),
            "last_lp_check": time.time(),
            "last_lp_block": w3_read.eth.block_number,
            "first_sell_block": w3_read.eth.block_number,
            "first_sell_detected": False,
        }


def queue_volume_check(
    pair_addr: str,
    token0: str,
    token1: str,
    passes: int,
    total: int,
    extra: dict,
    is_recheck: bool = False,
    attempt_num: int = None,
):
    if pair_addr not in volume_checks:
        volume_checks[pair_addr] = {
            "token0": token0,
            "token1": token1,
            "start": time.time(),
            "last_check": 0,
            "passes": passes,
            "total": total,
            "extra": extra,
            "is_recheck": is_recheck,
            "attempt": attempt_num,
        }


def check_marketcap_milestones(pair_addr: str, current_mc: float):
    if pair_addr not in passing_pairs:
        return
    info = passing_pairs[pair_addr]
    init_mc = info["initial_mc"]
    if init_mc <= 0:
        return
    ratio = current_mc / init_mc
    for milestone in [2, 3, 5, 10]:
        if ratio >= milestone and milestone not in info["mc_milestones_hit"]:
            info["mc_milestones_hit"].add(milestone)
            logger.info(
                f"[MC Milestone] {pair_addr} reached {milestone}x from ${init_mc:,.0f}"
            )
            send_telegram_message(
                f"[MC Milestone] {pair_addr} reached {milestone}x from ${init_mc:,.0f}"
            )


def handle_passing_refreshes():
    now_ts = time.time()
    remove_list = []
    for pair, data in list(passing_pairs.items()):
        if not data.get("first_sell_detected"):
            seller = detect_first_sell(
                pair,
                data["token0"],
                data["token1"],
                data.get("first_sell_block", w3_read.eth.block_number),
            )
            if seller:
                token_addr = get_non_weth_token(data["token0"], data["token1"])
                info, risk = analyze_seller_wallet(token_addr, seller)
                send_telegram_message(
                    f"[FirstSell] {pair} sold by <code>{seller}</code> ({info}, risk {risk})"
                )
                record_first_sell(pair, seller, w3_read.eth.block_number)
                data["first_sell_detected"] = True
                if risk >= 5:
                    remove_list.append(pair)

        idx = data["attempt_index"]

        # normal refresh attempts
        if idx < len(PASSING_REFRESH_DELAYS):
            delay = PASSING_REFRESH_DELAYS[idx]
            if (now_ts - data["last_attempt"]) >= delay:
                data["attempt_index"] += 1
                data["last_attempt"] = now_ts
                attempt_num = data["attempt_index"]
                logger.info(f"[Refresh] Attempt #{attempt_num} for {pair}")

                passes, total, xtra = recheck_logic_detail(
                    pair, data["token0"], data["token1"], attempt_num, True
                )
                fail, reason = critical_verification_failure(xtra)
                if fail:
                    logger.info(f"[Remove] {pair} removed: {reason}")
                    if reason not in ("verification error", "risk score 9999"):
                        send_telegram_message(f"[Remove] {pair} removed: {reason}")
                    remove_list.append(pair)
                    continue
                data["last_liquidity"] = xtra.get(
                    "liquidityUsd", data.get("last_liquidity")
                )
                data["last_liq_check"] = now_ts
                if passes < MIN_PASS_THRESHOLD:
                    logger.info(f"[Refresh] => removing {pair} from passing.")
                    remove_list.append(pair)
        else:
            # no more attempts
            if not data["no_more_attempts_logged"]:
                logger.info(f"[Refresh] No more scheduled attempts for {pair}")
                data["no_more_attempts_logged"] = True

        if pair in remove_list:
            continue

        # silent check for MC
        if data["silent_active"]:
            detection_time = detected_at.get(pair.lower(), data["last_attempt"])
            elapsed = now_ts - detection_time
            if elapsed > SILENT_CHECK_DURATION:
                data["silent_active"] = False
                logger.info(f"[SilentCheck] => ended for {pair}")
            else:
                if (now_ts - data["last_silent_check"]) >= SILENT_CHECK_INTERVAL:
                    logger.info(f"[SilentCheck] => 10-min for {pair}")
                    p2, t2, xtra = check_pair_criteria(
                        pair, data["token0"], data["token1"]
                    )
                    fail, reason = critical_verification_failure(xtra)
                    if fail:
                        logger.info(f"[Remove] {pair} removed: {reason}")
                        if reason not in ("verification error", "risk score 9999"):
                            send_telegram_message(f"[Remove] {pair} removed: {reason}")
                        remove_list.append(pair)
                        continue
                    cmc = xtra.get("marketCap", 0)
                    data["last_liquidity"] = xtra.get(
                        "liquidityUsd", data.get("last_liquidity")
                    )
                    data["last_liq_check"] = now_ts
                    pmc = data["last_silent_mc"]
                    if elapsed >= 1800 and abs(cmc - pmc) < 1e-5:
                        logger.info(
                            f"[SilentCheck] => Inactive after 30m no MC change {pair}"
                        )
                        data["silent_active"] = False
                    else:
                        if cmc > pmc:
                            data["last_silent_mc"] = cmc
                            data["no_new_high_count"] = 0
                            ratio = 0
                            if data["initial_mc"] > 0:
                                ratio = cmc / data["initial_mc"]
                            logger.info(
                                f"[SilentCheck] {pair} new MC high ~${cmc:,.0f} (x{ratio:.2f})"
                            )
                            send_telegram_message(
                                f"[SilentCheck] {pair} new MC high ~${cmc:,.0f} (x{ratio:.2f})"
                            )
                        else:
                            data["no_new_high_count"] += 1
                            if data["no_new_high_count"] >= 2:
                                logger.info(f"[SilentCheck] => done no new high {pair}")
                                data["silent_active"] = False
                    data["last_silent_check"] = now_ts

        # quick 30s honeypot check
        if pair not in remove_list:
            if (now_ts - data["last_hp_check"]) >= 30:
                data["last_hp_check"] = now_ts
                hp0 = check_honeypot_is(data["token0"], pair_addr=pair)
                hp1 = check_honeypot_is(data["token1"], pair_addr=pair)
                if hp0 or hp1:
                    logger.info(f"[HoneypotDetected] {pair} removed from passing pairs")
                    send_telegram_message(f"[HoneypotDetected] {pair} removed from passing pairs")
                    remove_list.append(pair)

        # liquidity removal check every 5m
        if pair not in remove_list:
            if (now_ts - data.get("last_liq_check", 0)) >= 300:
                ds = fetch_dexscreener_data(data["token0"], pair)
                if ds:
                    new_liq = ds.get("liquidityUsd", 0)
                    old_liq = data.get("last_liquidity", new_liq)
                    if old_liq and new_liq < old_liq * 0.9:
                        send_telegram_message(
                            f"[LiquidityRemoved] {pair} liquidity dropped from ${old_liq:,.0f} to ${new_liq:,.0f} => potential rugpull"
                        )
                        remove_list.append(pair)
                    data["last_liquidity"] = new_liq

                # check LP supply drop
                supply_now = get_lp_total_supply(pair)
                prev_supply = data.get("lp_supply")
                if (
                    supply_now is not None
                    and prev_supply
                    and supply_now < prev_supply * 0.9
                ):
                    send_telegram_message(
                        f"[LiquidityRemoved] {pair} LP supply drop detected"
                    )
                    remove_list.append(pair)
                if supply_now is not None:
                    data["lp_supply"] = supply_now

                # check for Burn events
                last_block = data.get("last_lp_block", w3_read.eth.block_number)
                try:
                    logs = w3_read.eth.get_logs(
                        {
                            "address": pair,
                            "fromBlock": last_block + 1,
                            "toBlock": "latest",
                            "topics": [BURN_TOPIC],
                        }
                    )
                    if logs:
                        send_telegram_message(
                            f"[LiquidityRemoved] {pair} burn event detected"
                        )
                        remove_list.append(pair)
                    data["last_lp_block"] = w3_read.eth.block_number
                except Exception as e:
                    logger.debug(f"burn log error: {e}")


                if check_recent_liquidity_removal(pair):
                    send_telegram_message(
                        f"[LiquidityRemoved] {pair} LP burn detected via Etherscan"
                    )
                    remove_list.append(pair)

                data["last_liq_check"] = now_ts

        if pair not in remove_list:
            # done attempts & silent => remove
            if (
                data["attempt_index"] >= len(PASSING_REFRESH_DELAYS)
                and not data["silent_active"]
            ):
                remove_list.append(pair)

    for rm in remove_list:
        passing_pairs.pop(rm, None)
        logger.info(f"[PassingPairs] => removed {rm}")


def handle_volume_checks():
    now = time.time()
    remove = []
    for pair, data in list(volume_checks.items()):
        if (now - data["last_check"]) >= 5 or (now - data["start"]) >= 300:
            data["last_check"] = now
            ds = fetch_dexscreener_data(data["token0"], pair)
            if not ds:
                if (now - data["start"]) >= 300:
                    remove.append(pair)
                continue
            vol = ds.get("volume24h", 0)
            trades = ds.get("buys", 0) + ds.get("sells", 0)
            if vol >= MIN_VOLUME_USD and trades >= MIN_TRADES_REQUIRED:
                extra = data["extra"]
                extra.update(ds)
                send_ui_criteria_message(
                    pair,
                    data["passes"],
                    data["total"],
                    is_recheck=data.get("is_recheck", False),
                    token_name=extra.get("tokenName", ""),
                    clog_percent=extra.get("clogPercent"),
                    extra_stats=extra,
                    recheck_attempt=data.get("attempt"),
                    is_passing_refresh=False,
                )
                queue_passing_refresh(
                    pair,
                    data["token0"],
                    data["token1"],
                    ds.get("marketCap", 0),
                    ds.get("liquidityUsd", 0),
                )
                remove.append(pair)
            elif (now - data["start"]) >= 300:
                logger.info(f"[VolumeCheck] {pair} failed to reach targets")
                remove.append(pair)

    for rm in remove:
        volume_checks.pop(rm, None)


def handle_wallet_updates():
    now = time.time()
    for token, data in list(WALLET_REPORT_CACHE.items()):
        if now - data.get("ts", 0) >= WALLET_REPORT_TTL:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    wallet_tracker.generate_wallet_report(token), wallet_event_loop
                )
                WALLET_REPORT_CACHE[token]["report"] = fut.result()
                WALLET_REPORT_CACHE[token]["ts"] = time.time()
                time.sleep(1)
            except Exception as e:
                logger.error(f"wallet update error for {token}: {e}")


###########################################################
# 10.5 recheck_logic_detail
###########################################################


def recheck_logic_detail(
    pair_addr: str, t0: str, t1: str, attempt_num: int, is_passing_refresh: bool
):
    passes, total, extra = check_pair_criteria(pair_addr, t0, t1)
    mode = "Refresh" if is_passing_refresh else "Recheck"
    logger.info(
        f"[{mode}] => {pair_addr} => {passes}/{total} passes (attempt {attempt_num}). "
        f"Verified={extra.get('contractCheckStatus')} Risk={extra.get('riskScore')}"
    )

    # evaluate for suspicious metrics before notifying
    fail_reasons = evaluate_fail_reasons(extra)
    if fail_reasons:
        passes = 0

    if passes >= MIN_PASS_THRESHOLD and is_passing_refresh:
        send_ui_criteria_message(
            pair_addr,
            passes,
            total,
            is_recheck=False,
            token_name=extra.get("tokenName", ""),
            clog_percent=extra.get("clogPercent"),
            extra_stats=extra,
            recheck_attempt=attempt_num,
            is_passing_refresh=True,
        )

    mc = extra.get("marketCap", 0)
    if mc > 0:
        check_marketcap_milestones(pair_addr, mc)
    main_token = get_non_weth_token(t0, t1)
    wallet_rep = get_wallet_report(main_token)
    gem = {}
    patterns = {}
    liq_now = extra.get("liquidityUsd", 0)
    risk_flags = extra.get("riskFlags", {})
    wl_bl = risk_flags.get("canBlacklist") or risk_flags.get("botWhitelist")
    if (
        (
            wallet_rep
            and wallet_rep["risk_assessment"].get("overall_risk", 100) <= MAX_GEM_RISK_SCORE
            and wallet_rep["marketing_analysis"].get("activity_score", 0) >= MIN_GEM_MARKETING_SCORE
        )
        or wl_bl
    ) and MIN_GEM_MARKETCAP_USD <= mc <= MAX_GEM_MARKETCAP_USD and liq_now >= MIN_GEM_LIQUIDITY_USD:
        gem = detect_bull_market_early_gems(main_token, pair_addr, wallet_rep)
        patterns = identify_bull_pump_patterns({**extra, "token": main_token, "riskFlags": risk_flags, "dexPaid": extra.get("dexPaid", False)})
        send_bull_insights(pair_addr, extra.get("tokenName", ""), gem, patterns)
    if passes >= MIN_PASS_THRESHOLD:
        start_wallet_monitor(main_token)
    return (passes, total, extra)


###########################################################
# 11. RECHECK FAILING
###########################################################

pending_rechecks: Dict[str, dict] = {}


def queue_recheck(pair_addr: str, token0: str, token1: str):
    if pair_addr not in pending_rechecks:
        pending_rechecks[pair_addr] = {
            "token0": token0,
            "token1": token1,
            "attempt_index": 0,
            "last_attempt": time.time(),
            "created": time.time(),
            "fail_count": 0,
        }


def handle_rechecks():
    now_ts = time.time()
    rm_list = []
    for pair, data in list(pending_rechecks.items()):
        idx = data["attempt_index"]
        if idx >= len(RECHECK_DELAYS):
            logger.info(f"[Recheck] => removing {pair}, no more attempts.")
            rm_list.append(pair)
            continue

        delay = RECHECK_DELAYS[idx]
        if (now_ts - data["last_attempt"]) >= delay:
            data["attempt_index"] += 1
            data["last_attempt"] = now_ts
            attempt_num = data["attempt_index"]
            logger.info(f"[Recheck] Attempt #{attempt_num} for {pair}")

            passes, total, extra = recheck_logic_detail(
                pair, data["token0"], data["token1"], attempt_num, False
            )
            fail, reason = critical_verification_failure(extra)
            if fail:
                logger.info(f"[Remove] {pair} removed: {reason}")
                if reason not in ("verification error", "risk score 9999"):
                    send_telegram_message(f"[Remove] {pair} removed: {reason}")
                rm_list.append(pair)
                continue
            if passes >= MIN_PASS_THRESHOLD:
                vol_now = extra.get("volume24h", 0)
                trades_now = extra.get("buys", 0) + extra.get("sells", 0)
                if vol_now >= MIN_VOLUME_USD and trades_now >= MIN_TRADES_REQUIRED:
                    logger.info(f"[Recheck] => removing {pair}, now passing.")
                    rm_list.append(pair)
                    mc_now = extra.get("marketCap", 0)
                    liq_now = extra.get("liquidityUsd", 0)
                    send_ui_criteria_message(
                        pair,
                        passes,
                        total,
                        is_recheck=True,
                        token_name=extra.get("tokenName", ""),
                        clog_percent=extra.get("clogPercent"),
                        extra_stats=extra,
                        recheck_attempt=attempt_num,
                        is_passing_refresh=False,
                    )
                    queue_passing_refresh(
                        pair, data["token0"], data["token1"], mc_now, liq_now
                    )
                    main_token = get_non_weth_token(data["token0"], data["token1"])
                    main_token = get_non_weth_token(data['token0'], data['token1'])
                    start_wallet_monitor(main_token)
                    continue
                else:
                    logger.info(f"[VolumeCheck] waiting on {pair}")
                    queue_volume_check(
                        pair,
                        data["token0"],
                        data["token1"],
                        passes,
                        total,
                        extra,
                        is_recheck=True,
                        attempt_num=attempt_num,
                    )
                    rm_list.append(pair)
                    continue
            else:
                line = (
                    f"- {extra.get('tokenName','Unnamed')} => {passes}/{total} (Attempt #{attempt_num}) "
                    f"Verified={extra.get('contractCheckStatus')} Risk={extra.get('riskScore')}"
                )
                add_failed_recheck_message(line)
                data["fail_count"] = data.get("fail_count", 0) + 1
                if data["fail_count"] >= 3:
                    logger.info(f"[Recheck] => removing {pair}, failed 3 times")
                    rm_list.append(pair)
                    continue

        # stop after 10 minutes of failures
        elapsed = now_ts - data.get("created", data["last_attempt"])
        if elapsed >= 600:
            logger.info(f"[Recheck] => removing {pair}, failed after 10m")
            rm_list.append(pair)

    for r in rm_list:
        pending_rechecks.pop(r, None)
        logger.info(f"[Recheck] => removed {r}")


###########################################################
# Extra mapping for resolving pair -> (token0, token1)
known_pairs: Dict[str, Tuple[str, str]] = {}

###########################################################
# 12. HANDLE NEW PAIR
###########################################################


def handle_new_pair(pair_addr: str, token0: str, token1: str):
    """Process a newly created pair and evaluate its criteria."""
    try:
        paddr = to_checksum_address(pair_addr)
        token0 = to_checksum_address(token0)
        token1 = to_checksum_address(token1)
        detected_at[paddr.lower()] = time.time()
        known_pairs[paddr.lower()] = (token0, token1)

        passes, total, extra = check_pair_criteria(paddr, token0, token1)
        if extra.get("dexscreener_missing"):
            logger.info(f"[Requeue] {paddr} missing DexScreener data")
            queue_recheck(paddr, token0, token1)
            return
        if not isinstance(extra, dict):
            logger.error(f"check_pair_criteria returned invalid data for {paddr}: {extra}")
            extra = {}
        if not extra or not extra.get("tokenName"):
            logger.debug(f"[Skip] {paddr} missing DexScreener token name")
            return
        fail, reason = critical_verification_failure(extra)
        if fail:
            logger.info(f"[Remove] {paddr} removed: {reason}")
            if reason not in ("verification error", "risk score 9999"):
                send_telegram_message(f"[Remove] {paddr} removed: {reason}")
            return
        logger.info(
            f"[NewPair] {paddr} => {passes}/{total} partial passes. Verified={extra.get('contractCheckStatus')} Risk={extra.get('riskScore')}"
        )
        store_pair_record(paddr, token0, token1, passes, total, extra)

        fail_reasons = evaluate_fail_reasons(extra)
        if fail_reasons:
            passes = 0

        mc_now = extra.get("marketCap", 0)
        if mc_now > 0:
            check_marketcap_milestones(paddr, mc_now)

        main_token = get_non_weth_token(token0, token1)
        wallet_report = extra.get("wallet_analysis")
        if not wallet_report or "risk_assessment" not in wallet_report:
            wallet_report = get_wallet_report(main_token)
        gem = {}
        patterns = {}
        liq_now = extra.get("liquidityUsd", 0)
        risk_flags = extra.get("riskFlags", {})
        wl_bl = risk_flags.get("canBlacklist") or risk_flags.get("botWhitelist")
        risk_score = wallet_report.get("risk_assessment", {}).get("overall_risk", 100)
        marketing_score = wallet_report.get("marketing_analysis", {}).get("activity_score", 0)
        risk_ok = risk_score <= MAX_GEM_RISK_SCORE
        marketing_ok = marketing_score >= MIN_GEM_MARKETING_SCORE
        near_risk = risk_score <= MAX_GEM_RISK_SCORE + GEM_WARNING_RISK_DELTA
        near_marketing = marketing_score >= MIN_GEM_MARKETING_SCORE - GEM_WARNING_MARKETING_DELTA
        mc_ok = MIN_GEM_MARKETCAP_USD <= mc_now <= MAX_GEM_MARKETCAP_USD
        liq_ok = liq_now >= MIN_GEM_LIQUIDITY_USD
        near_mc = MIN_GEM_MARKETCAP_USD * 0.8 <= mc_now < MIN_GEM_MARKETCAP_USD
        near_liq = MIN_GEM_LIQUIDITY_USD * 0.8 <= liq_now < MIN_GEM_LIQUIDITY_USD
        if ((risk_ok and marketing_ok and wallet_report) or wl_bl) and mc_ok and liq_ok:
            report_for_gem = wallet_report if risk_ok and marketing_ok else None
            gem = detect_bull_market_early_gems(main_token, paddr, report_for_gem)
            patterns = identify_bull_pump_patterns({**extra, "token": main_token, "riskFlags": risk_flags, "dexPaid": extra.get("dexPaid", False)})
            send_bull_insights(paddr, extra.get("tokenName", ""), gem, patterns)
        elif (
            (mc_ok and liq_ok and (near_risk or near_marketing or wl_bl))
            or ((near_mc or near_liq) and risk_ok and marketing_ok)
        ):
            send_borderline_warning(paddr, mc_now, liq_now, risk_score, marketing_score, bool(wl_bl))

        if mc_ok and liq_ok:
            enhanced_results = integrate_enhanced_detection(main_token, paddr, extra)
            if enhanced_results["action_required"]:
                send_enhanced_alert(paddr, enhanced_results)
            extra["enhanced_gem_score"] = enhanced_results["gem_analysis"]["score"]
            extra["enhanced_pattern_score"] = enhanced_results["pattern_analysis"]["score"]
            extra["bull_market_signals"] = enhanced_results
        else:
            extra["enhanced_gem_score"] = 0
            extra["enhanced_pattern_score"] = 0

        if passes >= MIN_PASS_THRESHOLD:
            start_wallet_monitor(main_token)

        if passes >= MIN_PASS_THRESHOLD:
            vol_now = extra.get("volume24h", 0)
            trades_now = extra.get("buys", 0) + extra.get("sells", 0)
            if vol_now >= MIN_VOLUME_USD and trades_now >= MIN_TRADES_REQUIRED:
                send_ui_criteria_message(
                    paddr,
                    passes,
                    total,
                    is_recheck=False,
                    token_name=extra.get("tokenName", ""),
                    clog_percent=extra.get("clogPercent"),
                    extra_stats=extra,
                )
                liq_now = extra.get("liquidityUsd", 0)
                queue_passing_refresh(paddr, token0, token1, mc_now, liq_now)
            else:
                logger.info(f"[VolumeCheck] waiting on {paddr}")
                queue_volume_check(paddr, token0, token1, passes, total, extra)
        else:
            logger.info("Not enough passes => queue recheck.")
            queue_recheck(paddr, token0, token1)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        line = tb[-1].lineno if tb else 'unknown'
        logger.error(
            f"handle_new_pair failed for {pair_addr} at line {line}: {e}",
        )
        logger.debug(''.join(traceback.format_tb(e.__traceback__)))


###########################################################
# 13. OPTIONAL REPORTS
###########################################################

_last_daily_date = None
_last_weekly_date = None
_last_hourly_pnl_time = 0




###########################################################
# 14. MAIN LOOP
###########################################################


def load_last_block(fname: str) -> int:
    if os.path.exists(fname):
        try:
            data = json.load(open(fname, "r"))
            return data.get("last_block", 0)
        except Exception:
            pass
    return 0


def save_last_block(bn: int, fname: str):
    try:
        json.dump({"last_block": bn}, open(fname, "w"))
    except Exception:
        pass


def main():
    logger.info("Starting advanced CryptoBot...")

    last_block_v2 = load_last_block(LAST_BLOCK_FILE_V2)
    if last_block_v2 == 0:
        last_block_v2 = safe_block_number(False)
        save_last_block(last_block_v2, LAST_BLOCK_FILE_V2)
        logger.info(f"Initialized v2 last_block={last_block_v2}")

    last_block_v3 = load_last_block(LAST_BLOCK_FILE_V3)
    if last_block_v3 == 0:
        last_block_v3 = safe_block_number(True)
        save_last_block(last_block_v3, LAST_BLOCK_FILE_V3)
        logger.info(f"Initialized v3 last_block={last_block_v3}")

    while True:
        try:
            curr_block_v2 = safe_block_number(False)
            if curr_block_v2 > last_block_v2:
                from_blk = last_block_v2 + 1
                to_blk = curr_block_v2

                filter_params = {
                    "fromBlock": from_blk,
                    "toBlock": to_blk,
                    "address": UNISWAP_V2_FACTORY_ADDRESS,
                    "topics": [PAIR_CREATED_TOPIC_V2],
                }
                logs = safe_get_logs(filter_params)

                for lg in logs:
                    if len(lg["topics"]) < 3:
                        continue
                    token0_hex = "0x" + lg["topics"][1].hex()[-40:]
                    token1_hex = "0x" + lg["topics"][2].hex()[-40:]
                    data_field = lg["data"]
                    if isinstance(data_field, HexBytes):
                        data_field = data_field.hex()
                    if data_field.startswith("0x"):
                        data_field = data_field[2:]
                    raw_bytes = bytes.fromhex(data_field)
                    try:
                        pair_addr, _ = decode(["address", "uint256"], raw_bytes)
                        handle_new_pair(pair_addr, token0_hex, token1_hex)
                    except Exception as e:
                        tb = traceback.extract_tb(e.__traceback__)
                        line = tb[-1].lineno if tb else 'unknown'
                        logger.error(
                            f"handle_new_pair v2 error at line {line}: {e}",
                        )

                last_block_v2 = to_blk
                save_last_block(last_block_v2, LAST_BLOCK_FILE_V2)
                logger.info(
                    f"Processed v2 blocks {from_blk}->{to_blk}, found {len(logs)} PairCreated logs"
                )

            curr_block_v3 = safe_block_number(True)
            if curr_block_v3 > last_block_v3:
                from_blk = last_block_v3 + 1
                to_blk = curr_block_v3

                filter_params = {
                    "fromBlock": from_blk,
                    "toBlock": to_blk,
                    "address": UNISWAP_V3_FACTORY_ADDRESS,
                    "topics": [POOL_CREATED_TOPIC_V3],
                }
                logs = safe_get_logs(filter_params, is_v3=True)

                for lg in logs:
                    if len(lg["topics"]) < 4:
                        continue
                    token0_hex = "0x" + lg["topics"][1].hex()[-40:]
                    token1_hex = "0x" + lg["topics"][2].hex()[-40:]
                    data_field = lg["data"]
                    if isinstance(data_field, HexBytes):
                        data_field = data_field.hex()
                    if data_field.startswith("0x"):
                        data_field = data_field[2:]
                    raw_bytes = bytes.fromhex(data_field)
                    try:
                        _, pool_addr = decode(["int24", "address"], raw_bytes)
                        handle_new_pair(pool_addr, token0_hex, token1_hex)
                    except Exception as e:
                        tb = traceback.extract_tb(e.__traceback__)
                        line = tb[-1].lineno if tb else 'unknown'
                        logger.error(
                            f"handle_new_pair v3 error at line {line}: {e}",
                        )

                last_block_v3 = to_blk
                save_last_block(last_block_v3, LAST_BLOCK_FILE_V3)
                logger.info(
                    f"Processed v3 blocks {from_blk}->{to_blk}, found {len(logs)} PoolCreated logs"
                )

            # handle failing rechecks
            handle_rechecks()

            # check volume milestones for pending pairs
            handle_volume_checks()

            # handle passing refresh
            handle_passing_refreshes()

            # update wallet reports
            handle_wallet_updates()

            # flush fail buffer
            maybe_flush_failed_rechecks()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt => stopping.")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(5)

        time.sleep(MAIN_LOOP_SLEEP)


if __name__ == "__main__":
    main()