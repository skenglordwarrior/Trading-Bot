"""
Advanced Wallet Tracking System for Developer & Marketing Analysis
==================================================================
This system tracks and analyzes developer, marketing, and team wallets to predict
project success and detect potential rugpulls or scams.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Set, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import asyncio
import requests
from web3 import Web3
import json
import time
from collections import defaultdict
import numpy as np
import logging

# Constants shared with ethereumbotv2
ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"
ETHERSCAN_CHAIN_ID = os.getenv("ETHERSCAN_CHAIN_ID", "1")
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


_ETHERSCAN_LOOKUPS_ENABLED = _env_flag("ENABLE_ETHERSCAN_LOOKUPS", True)
_ETHERSCAN_DISABLED_REASON: Optional[str] = None


def set_etherscan_lookup_enabled(enabled: bool, reason: str = ""):
    """Allow the main bot to toggle Etherscan-dependent features."""
    global _ETHERSCAN_LOOKUPS_ENABLED, _ETHERSCAN_DISABLED_REASON
    if _ETHERSCAN_LOOKUPS_ENABLED != enabled:
        _ETHERSCAN_LOOKUPS_ENABLED = enabled
        if not enabled and reason:
            _ETHERSCAN_DISABLED_REASON = reason
            logger.warning("Disabling wallet tracker Etherscan lookups: %s", reason)


def _etherscan_enabled() -> bool:
    return _ETHERSCAN_LOOKUPS_ENABLED


async def _etherscan_get_async(params: dict, timeout: int = 20) -> dict:
    """Shared helper to perform an async Etherscan request.

    The wallet tracker module historically relied on the main bot to provide
    this helper.  Adding it locally keeps the tracker self-contained so it can
    be reused by stand-alone tools (like manual safety check scripts) without
    monkey patching.
    """

    if not _etherscan_enabled():
        return {
            "status": "0",
            "message": _ETHERSCAN_DISABLED_REASON or "etherscan lookups disabled",
            "result": [],
        }

    try:
        loop = asyncio.get_running_loop()

        def _do_request():
            resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()

        return await loop.run_in_executor(None, _do_request)
    except (requests.RequestException, asyncio.TimeoutError, OSError) as exc:
        set_etherscan_lookup_enabled(False, f"etherscan request failed: {exc}")
        return {"status": "0", "message": str(exc), "result": []}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unexpected Etherscan error: %s", exc)
        return {"status": "0", "message": str(exc), "result": []}

# ------------------------------------------------------------------
# Patch A: notifier injection so this module doesn't depend on a
# global send_telegram_message defined elsewhere.
# ------------------------------------------------------------------
_notifier_sync: Optional[Callable[[str], None]] = None
_notifier_async: Optional[Callable[[str], Awaitable[None]]] = None

def set_notifier(sync_fn: Optional[Callable[[str], None]] = None,
                 async_fn: Optional[Callable[[str], Awaitable[None]]] = None):
    """Inject notification functions from the main bot (sync or async)."""
    global _notifier_sync, _notifier_async
    _notifier_sync = sync_fn
    _notifier_async = async_fn

async def _notify(text: str):
    if _notifier_async is not None:
        await _notifier_async(text)
    elif _notifier_sync is not None:
        _notifier_sync(text)
    else:
        logger.info(text)


# Shared tracker instance to avoid repeated instantiation
_SHARED_TRACKER: Optional["SmartWalletTracker"] = None


def get_shared_tracker(w3_read=None, etherscan_key_getter=None):
    """Return a module-wide SmartWalletTracker instance."""
    global _SHARED_TRACKER
    if _SHARED_TRACKER is None:
        if w3_read is None or etherscan_key_getter is None:
            raise ValueError("Tracker not initialized")
        _SHARED_TRACKER = SmartWalletTracker(w3_read, etherscan_key_getter)
    return _SHARED_TRACKER

class WalletType(Enum):
    DEVELOPER = "developer"
    MARKETING = "marketing"
    TEAM = "team"
    LIQUIDITY = "liquidity"
    TREASURY = "treasury"
    PRESALE = "presale"
    UNKNOWN = "unknown"

@dataclass
class WalletActivity:
    address: str
    wallet_type: WalletType
    token_address: Optional[str]
    token_balance: Optional[float]
    eth_balance: float
    last_activity: int
    total_sells: int
    total_buys: int
    marketing_spends: List[dict]
    suspicious_activities: List[str]
    risk_score: int

class SmartWalletTracker:
    """Advanced wallet tracking system for project analysis"""
    
    def __init__(self, w3_read, etherscan_key_getter):
        self.w3 = w3_read
        self._etherscan_key_getter = etherscan_key_getter
        self.tracked_wallets: Dict[str, Dict[str, WalletActivity]] = {}
        self.marketing_spend_patterns = self.load_marketing_patterns()
        # Cache for previously generated reports to avoid API spam
        self.report_cache: Dict[str, Tuple[dict, float]] = {}
        self.report_ttl = 600  # seconds

    def get_etherscan_key(self) -> Optional[str]:
        if not _etherscan_enabled():
            return ""
        return self._etherscan_key_getter()
        
    def load_marketing_patterns(self) -> dict:
        """Load known marketing wallet patterns"""
        return {
            "dextools_payments": [
                "0x997Cc123cF292F46E55E6E63e806CD77714DB70f",  # DexTools payment wallet
                "0xfbfeaf0da0f2fde5c66df570133ae35f3eb58c9a",  # DexTools trending
            ],
            "twitter_promotion": [
                # Known crypto influencer wallets (anonymized)
                "0x8894e0a0c962cb723c1976a4421c95949be2d4e3",  # Example influencer
            ],
            "telegram_services": [
                "0xd4b69e8d62c880e9dd55d419d5e07435c3538342",  # Call channel services
            ],
            "cex_listings": [
                "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",  # Binance
                "0x0681d8db095565fe8a346fa0277bffde9c0edbbf",  # Huobi
            ]
        }
    
    async def identify_project_wallets(self, token_addr: str) -> Dict[str, WalletActivity]:
        """Identify and classify all project-related wallets"""
        wallets = {}

        # 0. Add contract creator as a developer wallet
        creator = await self.fetch_contract_creator(token_addr)
        if creator:
            activity = await self.analyze_wallet_activity(creator, token_addr, WalletType.DEVELOPER)
            wallets[creator.lower()] = activity

        # 1. Parse contract source for hardcoded addresses
        contract_wallets = await self.extract_wallets_from_contract(token_addr)

        # 2. Analyze initial token distribution
        distribution_wallets = await self.analyze_token_distribution(token_addr)

        # 3. Find wallets from events
        event_wallets = await self.find_wallets_from_events(token_addr)

        # 4. Classify each wallet
        all_wallets = {**contract_wallets, **distribution_wallets, **event_wallets}

        for addr, info in all_wallets.items():
            wallet_type = await self.classify_wallet(addr, token_addr, info)
            activity = await self.analyze_wallet_activity(addr, token_addr, wallet_type)
            wallets[addr] = activity

        return wallets

    async def fetch_contract_creator(self, token_addr: str) -> Optional[str]:
        """Fetch contract deployer using Etherscan."""
        if not _etherscan_enabled():
            return None
        api_key = self.get_etherscan_key()
        if not api_key:
            return None
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": token_addr,
            "apikey": api_key,
        }
        try:
            data = await _etherscan_get_async(params)
            if data.get("status") == "1" and data.get("result"):
                return data["result"][0].get("contractCreator")
        except Exception as e:
            logger.error(f"fetch_contract_creator error: {e}")
        return None

    async def fetch_contract_source(self, token_addr: str) -> dict:
        """Fetch verified contract source from Etherscan."""
        if not _etherscan_enabled():
            return {"status": "error", "source": []}
        token_addr = token_addr.lower()
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "contract",
            "action": "getsourcecode",
            "address": token_addr,
            "apikey": self.get_etherscan_key(),
        }
        try:
            data = await _etherscan_get_async(params)
            status = data.get("status")
            result = data.get("result", [])
            if status == "1" and result:
                scode = result[0].get("SourceCode", "")
                cname = result[0].get("ContractName", "")
                if not scode:
                    return {"status": "unverified", "source": []}
                sources_list = []
                stripped = scode.strip()
                if stripped.startswith("{"):
                    try:
                        parsed = json.loads(scode)
                        for fname, info in parsed.get("sources", {}).items():
                            content = info.get("content", "")
                            sources_list.append({"filename": fname, "content": content})
                    except Exception:
                        sources_list.append({"filename": cname or "contract.sol", "content": scode})
                else:
                    sources_list.append({"filename": cname or "contract.sol", "content": scode})
                return {"status": "verified", "source": sources_list}
            return {"status": "error", "source": []}
        except Exception as e:
            logger.error(f"fetch_contract_source error: {e}")
            return {"status": "error", "source": []}

    async def extract_wallets_from_contract(self, token_addr: str) -> Dict[str, dict]:
        """Extract wallet addresses from contract source code"""
        source_data = await self.fetch_contract_source(token_addr)
        if source_data.get("status") != "verified":
            return {}
            
        wallets = {}
        combined_source = "".join(part["content"] for part in source_data["source"])
        
        # Common patterns for wallet declarations
        patterns = [
            # address public marketingWallet = 0x...
            r'address\s+(?:public\s+)?(\w+Wallet|\w+Address)\s*=\s*(?:address\()?0x([a-fA-F0-9]{40})',
            # _marketingWallet = 0x...
            r'_(\w+Wallet|\w+Address)\s*=\s*(?:address\()?0x([a-fA-F0-9]{40})',
            # constructor params
            r'constructor\s*\([^)]*address\s+(\w+)[^)]*\)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, combined_source, re.IGNORECASE)
            for match in matches:
                var_name = match.group(1).lower()
                if len(match.groups()) > 1:
                    addr = "0x" + match.group(2)
                    wallet_type = self.infer_wallet_type_from_name(var_name)
                    wallets[addr.lower()] = {"type": wallet_type, "source": "contract"}
                    
        return wallets
    
    def infer_wallet_type_from_name(self, var_name: str) -> WalletType:
        """Infer wallet type from variable name"""
        var_lower = var_name.lower()
        
        if any(x in var_lower for x in ["market", "promo", "advert"]):
            return WalletType.MARKETING
        elif any(x in var_lower for x in ["dev", "team", "owner"]):
            return WalletType.DEVELOPER
        elif any(x in var_lower for x in ["liquid", "lp"]):
            return WalletType.LIQUIDITY
        elif any(x in var_lower for x in ["treasury", "reserve"]):
            return WalletType.TREASURY
        elif any(x in var_lower for x in ["presale", "sale"]):
            return WalletType.PRESALE
        else:
            return WalletType.UNKNOWN
    
    async def analyze_token_distribution(self, token_addr: str) -> Dict[str, dict]:
        """Analyze initial token distribution to find team wallets"""
        if not _etherscan_enabled():
            return {}
        wallets = {}

        # Get token creation block
        creation_tx = await self.get_contract_creation_tx(token_addr)
        if not creation_tx:
            return wallets

        # Query initial transfers
        api_key = self.get_etherscan_key()
        if not api_key:
            return wallets
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "account",
            "action": "tokentx",
            "contractaddress": token_addr,
            "startblock": creation_tx["blockNumber"],
            "endblock": int(creation_tx["blockNumber"]) + 1000,  # First ~1000 blocks
            "sort": "asc",
            "apikey": api_key,
        }

        try:
            data = await _etherscan_get_async(params)

            if data.get("status") != "1":
                return wallets

            # Analyze distribution pattern
            initial_receivers = defaultdict(float)
            total_supply = await self.get_token_total_supply(token_addr)

            for tx in data.get("result", [])[:50]:  # First 50 transfers
                to_addr = tx.get("to", "").lower()
                value = int(tx.get("value", "0"))

                if to_addr and to_addr != ZERO_ADDRESS.lower():
                    initial_receivers[to_addr] += value

            # Classify based on allocation percentage
            for addr, amount in initial_receivers.items():
                percentage = (amount / total_supply) * 100 if total_supply > 0 else 0

                if percentage > 5:  # Significant allocation
                    wallet_type = await self.classify_by_behavior(addr, token_addr)
                    wallets[addr] = {"type": wallet_type, "allocation": percentage}

        except Exception as e:
            logger.error(f"Distribution analysis error: {e}")

        return wallets
    
    async def classify_wallet(self, wallet_addr: str, token_addr: str, info: dict) -> WalletType:
        """Classify wallet based on multiple factors"""
        # If already classified from source
        if info.get("type") and info["type"] != WalletType.UNKNOWN:
            return info["type"]
            
        # Check if it's a known marketing service
        for service, addresses in self.marketing_spend_patterns.items():
            if wallet_addr in addresses:
                return WalletType.MARKETING
                
        # Analyze on-chain behavior
        return await self.classify_by_behavior(wallet_addr, token_addr)
    
    async def classify_by_behavior(self, wallet_addr: str, token_addr: str) -> WalletType:
        """Classify wallet based on transaction behavior"""
        if not _etherscan_enabled():
            return WalletType.UNKNOWN
        # Get recent transactions
        api_key = self.get_etherscan_key()
        if not api_key:
            return WalletType.UNKNOWN
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": wallet_addr,
            "sort": "desc",
            "page": 1,
            "offset": 100,
            "apikey": api_key,
        }

        try:
            data = await _etherscan_get_async(params)

            if data.get("status") != "1":
                return WalletType.UNKNOWN

            transactions = data.get("result", [])

            # Analyze patterns
            marketing_txs = 0
            dev_txs = 0

            for tx in transactions:
                to_addr = tx.get("to", "").lower()

                # Check if sending to known marketing addresses
                for addresses in self.marketing_spend_patterns.values():
                    if to_addr in [a.lower() for a in addresses]:
                        marketing_txs += 1

                # Check for contract deployments (dev activity)
                if not tx.get("to"):  # Contract creation
                    dev_txs += 1

            # Classify based on behavior
            if marketing_txs > 5:
                return WalletType.MARKETING
            elif dev_txs > 2:
                return WalletType.DEVELOPER
            else:
                return WalletType.TEAM

        except Exception as e:
            logger.error(f"Behavior classification error: {e}")
            return WalletType.UNKNOWN
    
    async def analyze_wallet_activity(
        self,
        wallet_addr: str,
        token_addr: Optional[str],
        wallet_type: WalletType,
    ) -> WalletActivity:
        """Comprehensive wallet activity analysis"""
        activity = WalletActivity(
            address=wallet_addr,
            wallet_type=wallet_type,
            token_address=token_addr,
            token_balance=None,
            eth_balance=0,
            last_activity=0,
            total_sells=0,
            total_buys=0,
            marketing_spends=[],
            suspicious_activities=[],
            risk_score=0
        )

        # Get current balances
        activity.eth_balance = await self.get_eth_balance(wallet_addr)
        token_txs: List[dict] = []
        if token_addr:
            activity.token_balance = await self.get_token_balance(wallet_addr, token_addr)

            # Analyze token transactions
            token_txs = await self.get_token_transactions(wallet_addr, token_addr)

            for tx in token_txs:
                if tx["from"].lower() == wallet_addr.lower():
                    activity.total_sells += 1
                    # Check if it's a dump
                    if self.is_suspicious_sell(tx, activity.token_balance or 0.0):
                        activity.suspicious_activities.append(
                            f"Large sell: {tx['value']} tokens"
                        )
                else:
                    activity.total_buys += 1

            if token_txs:
                try:
                    latest_ts = max(int(tx.get("timeStamp", 0)) for tx in token_txs)
                    activity.last_activity = max(activity.last_activity, latest_ts)
                except (TypeError, ValueError):
                    pass

        # For marketing wallets, track spending
        if wallet_type == WalletType.MARKETING:
            activity.marketing_spends = await self.track_marketing_spends(wallet_addr)

        if activity.marketing_spends:
            try:
                latest_spend = max(int(spend.get("timestamp", 0)) for spend in activity.marketing_spends)
                activity.last_activity = max(activity.last_activity, latest_spend)
            except (TypeError, ValueError):
                pass

        # Calculate risk score
        activity.risk_score = self.calculate_wallet_risk_score(activity)

        return activity
    
    async def track_marketing_spends(self, wallet_addr: str) -> List[dict]:
        """Track marketing wallet spending patterns"""
        if not _etherscan_enabled():
            return []
        spends = []

        # Get ETH transactions
        api_key = self.get_etherscan_key()
        if not api_key:
            return spends
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "account",
            "action": "txlist",
            "address": wallet_addr,
            "sort": "desc",
            "apikey": api_key,
        }

        try:
            data = await _etherscan_get_async(params)

            for tx in data.get("result", []):
                if tx["from"].lower() == wallet_addr.lower() and int(tx["value"]) > 0:
                    spend_type = self.identify_spend_type(tx["to"])
                    spends.append({
                        "to": tx["to"],
                        "value_eth": float(Web3.from_wei(int(tx["value"]), "ether")),
                        "timestamp": int(tx["timeStamp"]),
                        "type": spend_type,
                        "tx_hash": tx["hash"]
                    })

        except Exception as e:
            logger.error(f"Marketing spend tracking error: {e}")

        return spends
    
    def identify_spend_type(self, to_address: str) -> str:
        """Identify what type of marketing spend this is"""
        to_lower = to_address.lower()
        
        for spend_type, addresses in self.marketing_spend_patterns.items():
            if to_lower in [a.lower() for a in addresses]:
                return spend_type
                
        # Check if it's a known DEX router (buying back tokens)
        dex_routers = [
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
            "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3
        ]
        
        if to_lower in dex_routers:
            return "dex_interaction"
            
        return "unknown"
    
    def calculate_wallet_risk_score(self, activity: WalletActivity) -> int:
        """Calculate risk score for a wallet (0-100)"""
        risk_score = 0
        
        # High sell pressure
        if activity.total_sells > activity.total_buys * 2:
            risk_score += 20
            
        # Low ETH balance (can't pay for marketing)
        if activity.wallet_type == WalletType.MARKETING and activity.eth_balance < 0.5:
            risk_score += 30
            
        # No recent activity
        if activity.last_activity and (time.time() - activity.last_activity) > 86400 * 7:  # 7 days
            risk_score += 15
            
        # Suspicious activities
        risk_score += len(activity.suspicious_activities) * 10
        
        # Positive signals (reduce risk)
        if activity.wallet_type == WalletType.MARKETING:
            # Active marketing spending
            recent_spends = [s for s in activity.marketing_spends 
                           if time.time() - s["timestamp"] < 86400 * 3]  # Last 3 days
            if len(recent_spends) > 3:
                risk_score -= 20
                
        return max(0, min(100, risk_score))
    
    def is_suspicious_sell(self, tx: dict, current_balance: float) -> bool:
        """Check if a sell transaction is suspicious"""
        sell_amount = float(tx.get("value", 0))
        
        # Selling more than 10% of holdings at once
        if current_balance > 0 and sell_amount > current_balance * 0.1:
            return True
            
        # Large absolute sells (need to adjust based on token)
        # This is a placeholder - should be dynamic based on token metrics
        if sell_amount > 1000000:  # Example threshold
            return True
            
        return False
    
    
    async def monitor_wallet_realtime(
        self,
        token_addr: str,
        callback,
        stop_event: Optional[asyncio.Event] = None,
    ):
        """Real-time monitoring of project wallets with clean cancellation support."""
        if token_addr not in self.tracked_wallets:
            self.tracked_wallets[token_addr] = await self.identify_project_wallets(token_addr)
        try:
            while not (stop_event and stop_event.is_set()):
                try:
                    for wallet_addr, old_activity in list(self.tracked_wallets[token_addr].items()):
                        # Get updated activity
                        new_activity = await self.analyze_wallet_activity(
                            wallet_addr, token_addr, old_activity.wallet_type
                        )
                        # Check for significant changes
                        if self.has_significant_change(old_activity, new_activity):
                            await callback(token_addr, wallet_addr, old_activity, new_activity)
                        # Update tracked data
                        self.tracked_wallets[token_addr][wallet_addr] = new_activity
                    # Sleep in small chunks so stop_event is responsive
                    for _ in range(60):
                        if stop_event and stop_event.is_set():
                            break
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info(f"Wallet monitoring cancelled for {token_addr}")
                    break
                except Exception as e:
                    logger.error(f"Wallet monitoring error: {e}")
                    # Backoff but remain cancellable
                    for _ in range(60):
                        if stop_event and stop_event.is_set():
                            break
                        await asyncio.sleep(1)
        finally:
            logger.info(f"Wallet monitoring stopped for {token_addr}")

    def has_significant_change(self, old: WalletActivity, new: WalletActivity) -> bool:
        """Detect significant changes in wallet activity"""
        # Major sell
        if new.total_sells > old.total_sells:
            return True

        # Large ETH withdrawal from marketing wallet
        if old.wallet_type == WalletType.MARKETING:
            if old.eth_balance - new.eth_balance > 1.0:  # More than 1 ETH withdrawn
                return True

        # Risk score increase
        if new.risk_score - old.risk_score > 20:
            return True

        return False

    async def generate_wallet_report(self, token_addr: str) -> dict:
        """Generate comprehensive wallet analysis report"""
        token_addr = token_addr.lower()
        now = time.time()
        cached = self.report_cache.get(token_addr)
        if cached and now - cached[1] < self.report_ttl:
            return cached[0]

        wallets = await self.identify_project_wallets(token_addr)

        report = {
            "token": token_addr,
            "timestamp": int(now),
            "wallet_summary": {
                "total_wallets": len(wallets),
                "developer_wallets": len([w for w in wallets.values() if w.wallet_type == WalletType.DEVELOPER]),
                "marketing_wallets": len([w for w in wallets.values() if w.wallet_type == WalletType.MARKETING]),
            },
            "risk_assessment": {
                "overall_risk": 0,
                "red_flags": [],
                "positive_signals": []
            },
            "marketing_analysis": {
                "total_spend_eth": 0,
                "spend_categories": defaultdict(float),
                "activity_score": 0
            },
            "developer_analysis": {
                "holding_percentage": 0,
                "sell_pressure": 0,
                "commitment_score": 0
            }
        }
        
        # Analyze each wallet
        total_token_balance = 0
        dev_token_balance = 0
        
        for wallet in wallets.values():
            if wallet.token_balance is not None:
                total_token_balance += wallet.token_balance

                # Developer analysis
                if wallet.wallet_type == WalletType.DEVELOPER:
                    dev_token_balance += wallet.token_balance
                    if wallet.total_sells > 5:
                        report["risk_assessment"]["red_flags"].append(
                            f"Dev wallet {wallet.address[:8]}... has {wallet.total_sells} sells"
                        )

            if wallet.token_balance is None and wallet.wallet_type == WalletType.DEVELOPER and wallet.total_sells > 5:
                report["risk_assessment"]["red_flags"].append(
                    f"Dev wallet {wallet.address[:8]}... has {wallet.total_sells} sells"
                )

            # Marketing analysis
            if wallet.wallet_type == WalletType.MARKETING:
                for spend in wallet.marketing_spends:
                    value_eth = float(spend["value_eth"])
                    report["marketing_analysis"]["total_spend_eth"] += value_eth
                    report["marketing_analysis"]["spend_categories"][spend["type"]] += value_eth

                # Check if actively marketing
                recent_spends = [s for s in wallet.marketing_spends
                               if time.time() - s["timestamp"] < 86400 * 7]  # Last week

                if len(recent_spends) > 5:
                    report["risk_assessment"]["positive_signals"].append(
                        "Active marketing spending detected"
                    )
                    
        # Calculate scores
        if total_token_balance > 0:
            report["developer_analysis"]["holding_percentage"] = (dev_token_balance / total_token_balance) * 100
            
        # Marketing activity score (0-100)
        marketing_score = min(100, report["marketing_analysis"]["total_spend_eth"] * 10)
        report["marketing_analysis"]["activity_score"] = marketing_score

        # Flag if project paid for Dex platform promotion
        dex_paid = report["marketing_analysis"]["spend_categories"].get("dextools_payments", 0) > 0
        report["marketing_analysis"]["dex_paid"] = dex_paid
        if dex_paid:
            report["marketing_analysis"]["activity_score"] = min(
                100, report["marketing_analysis"]["activity_score"] + 20
            )
        
        # Overall risk
        risk_scores = [w.risk_score for w in wallets.values()]
        report["risk_assessment"]["overall_risk"] = np.mean(risk_scores) if risk_scores else 50
        self.report_cache[token_addr] = (report, now)
        return report
    
    # Helper methods
    async def get_eth_balance(self, wallet_addr: str) -> float:
        """Get ETH balance of wallet"""
        try:
            balance_wei = self.w3.eth.get_balance(Web3.to_checksum_address(wallet_addr))
            return float(Web3.from_wei(balance_wei, "ether"))
        except Exception:
            return 0.0
            
    async def get_token_balance(self, wallet_addr: str, token_addr: Optional[str]) -> float:
        """Get token balance of wallet"""
        if not token_addr:
            return 0.0
        try:
            abi = [{"constant": True, "inputs": [{"name": "", "type": "address"}],
                   "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                   "type": "function"}]
            contract = self.w3.eth.contract(Web3.to_checksum_address(token_addr), abi=abi)
            balance = contract.functions.balanceOf(Web3.to_checksum_address(wallet_addr)).call()
            
            # Get decimals
            decimals_abi = [{"constant": True, "inputs": [], "name": "decimals", 
                           "outputs": [{"name": "", "type": "uint8"}], "type": "function"}]
            decimals_contract = self.w3.eth.contract(Web3.to_checksum_address(token_addr), abi=decimals_abi)
            decimals = decimals_contract.functions.decimals().call()
            
            return balance / (10 ** decimals)
        except Exception:
            return 0.0
            
    async def get_contract_creation_tx(self, contract_addr: str) -> Optional[dict]:
        """Get contract creation transaction"""
        if not _etherscan_enabled():
            return None
        api_key = self.get_etherscan_key()
        if not api_key:
            return None
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "contract",
            "action": "getcontractcreation",
            "contractaddresses": contract_addr,
            "apikey": api_key,
        }

        try:
            data = await _etherscan_get_async(params)

            if data.get("status") == "1" and data.get("result"):
                return data["result"][0]
        except Exception:
            logger.exception("contract creation lookup failed")

        return None
        
    async def get_token_total_supply(self, token_addr: str) -> float:
        """Get total supply of token"""
        try:
            abi = [{"constant": True, "inputs": [], "name": "totalSupply", 
                   "outputs": [{"name": "", "type": "uint256"}], "type": "function"}]
            contract = self.w3.eth.contract(Web3.to_checksum_address(token_addr), abi=abi)
            return float(contract.functions.totalSupply().call())
        except Exception:
            return 0.0
            
    async def get_token_transactions(self, wallet_addr: str, token_addr: Optional[str]) -> List[dict]:
        """Get token transactions for a wallet"""
        if not token_addr:
            return []
        if not _etherscan_enabled():
            return []
        api_key = self.get_etherscan_key()
        if not api_key:
            return []
        params = {
            "chainid": ETHERSCAN_CHAIN_ID,
            "module": "account",
            "action": "tokentx",
            "address": wallet_addr,
            "contractaddress": token_addr,
            "sort": "desc",
            "apikey": api_key,
        }

        try:
            data = await _etherscan_get_async(params)

            if data.get("status") == "1":
                return data.get("result", [])
        except Exception:
            logger.exception("token transaction lookup failed")

        return []
    
    async def find_wallets_from_events(self, token_addr: str) -> Dict[str, dict]:
        """Find wallets from contract events"""
        # This would parse events like OwnershipTransferred, MarketingWalletUpdated, etc.
        # Placeholder for now
        return {}


# Integration with main bot
async def wallet_activity_callback(token_addr: str, wallet_addr: str, old: WalletActivity, new: WalletActivity):
    """Callback for significant wallet activity"""
    alert_msg = f"🚨 <b>Wallet Activity Alert</b>\n"
    alert_msg += f"Token: <code>{token_addr}</code>\n"
    alert_msg += f"Wallet: <code>{wallet_addr}</code> ({new.wallet_type.value})\n\n"
    
    # Check what changed
    if new.total_sells > old.total_sells:
        sell_count = new.total_sells - old.total_sells
        alert_msg += f"⚠️ New sells detected: {sell_count} transactions\n"
        
    if old.eth_balance - new.eth_balance > 0.5:
        withdrawn = old.eth_balance - new.eth_balance
        alert_msg += f"💸 ETH withdrawn: {withdrawn:.2f} ETH\n"
        
    if new.wallet_type == WalletType.MARKETING:
        recent_spends = [s for s in new.marketing_spends if s not in old.marketing_spends]
        if recent_spends:
            alert_msg += f"📣 New marketing spends: {len(recent_spends)} transactions\n"
            for spend in recent_spends[:3]:  # Show first 3
                alert_msg += f"  • {spend['value_eth']:.3f} ETH to {spend['type']}\n"
                
    alert_msg += f"\nRisk Score: {old.risk_score} → {new.risk_score}"
    
    await _notify(alert_msg)


def integrate_wallet_tracking(handle_new_pair_func):
    """Decorator to add wallet tracking to handle_new_pair"""
    async def wrapped(pair_addr: str, token0: str, token1: str):
        # Run original function
        await handle_new_pair_func(pair_addr, token0, token1)
        
        # Add wallet tracking
        main_token = get_non_weth_token(token0, token1)
        tracker = get_shared_tracker(w3_read, get_next_etherscan_key)

        # Generate initial report
        report = await tracker.generate_wallet_report(main_token)
        
        # Send report if concerning
        if report["risk_assessment"]["overall_risk"] > 70:
            msg = f"⚠️ <b>High Risk Wallet Configuration</b>\n"
            msg += f"Token: <code>{main_token}</code>\n"
            msg += f"Overall Risk: {report['risk_assessment']['overall_risk']:.0f}/100\n"
            msg += f"Red Flags: {', '.join(report['risk_assessment']['red_flags'])}\n"
            await _notify(msg)
            
        elif report["marketing_analysis"]["activity_score"] > 50:
            msg = f"✅ <b>Active Marketing Detected</b>\n"
            msg += f"Token: <code>{main_token}</code>\n"
            msg += f"Marketing Spend: {report['marketing_analysis']['total_spend_eth']:.2f} ETH\n"
            msg += f"Activity Score: {report['marketing_analysis']['activity_score']:.0f}/100\n"
            await _notify(msg)
            
        # Start monitoring (non-blocking)
        asyncio.create_task(
            tracker.monitor_wallet_realtime(main_token, wallet_activity_callback)
        )
        
    return wrapped


# Add to check_pair_criteria for wallet risk assessment
def check_pair_criteria_with_wallets(
    pair_addr: str,
    token0: str,
    token1: str,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[int, int, dict]:
    """Enhanced criteria check including wallet analysis"""
    # Run original checks
    passes, total, extra = check_pair_criteria(pair_addr, token0, token1)

    # Add wallet analysis using shared tracker
    main_token = get_non_weth_token(token0, token1)
    tracker = get_shared_tracker(w3_read, get_next_etherscan_key)

    # Run wallet analysis without blocking
    fut = asyncio.run_coroutine_threadsafe(
        tracker.generate_wallet_report(main_token), loop
    )
    report = fut.result()
    
    # Add wallet metrics to criteria
    total += 2  # Add 2 more checks
    
    # Check 1: Low wallet risk
    if report["risk_assessment"]["overall_risk"] < 50:
        passes += 1
        
    # Check 2: Active marketing
    if report["marketing_analysis"]["activity_score"] > 30:
        passes += 1
        
    # Add wallet data to extra
    extra["wallet_analysis"] = {
        "risk_score": report["risk_assessment"]["overall_risk"],
        "marketing_score": report["marketing_analysis"]["activity_score"],
        "marketing_spend_eth": report["marketing_analysis"]["total_spend_eth"],
        "dev_holding_percentage": report["developer_analysis"]["holding_percentage"],
        "red_flags": report["risk_assessment"]["red_flags"],
        "positive_signals": report["risk_assessment"]["positive_signals"]
    }

    return passes, total, extra