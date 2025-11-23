"""Maestro execution client for automated buys.

This module wraps the Maestro HTTP API with retry/back-off and exposes a
non-blocking interface that can be called from synchronous scanning code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import aiohttp


logger = logging.getLogger(__name__)


def _parse_decimal(value: str, default: Decimal = Decimal(0)) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return default


@dataclass
class MaestroConfig:
    api_base_url: str = os.getenv("MAESTRO_API_BASE_URL", "").rstrip("/")
    api_key: str = os.getenv("MAESTRO_API_KEY", "")
    account: str = os.getenv("MAESTRO_ACCOUNT", "")
    buy_amount_eth: Decimal = _parse_decimal(os.getenv("MAESTRO_BUY_AMOUNT_ETH", "0"))
    slippage_bps: int = int(os.getenv("MAESTRO_SLIPPAGE_BPS", "75"))
    priority_fee_gwei: Optional[float] = (
        float(os.getenv("MAESTRO_PRIORITY_FEE_GWEI", "0"))
        if os.getenv("MAESTRO_PRIORITY_FEE_GWEI")
        else None
    )
    enabled: bool = os.getenv("MAESTRO_ENABLED", "false").lower() == "true"
    dry_run: bool = os.getenv("MAESTRO_DRY_RUN", "true").lower() == "true"
    max_retries: int = int(os.getenv("MAESTRO_MAX_RETRIES", "2"))
    retry_backoff: float = float(os.getenv("MAESTRO_RETRY_BACKOFF", "0.75"))
    request_timeout: float = float(os.getenv("MAESTRO_REQUEST_TIMEOUT", "15"))
    trade_path: str = os.getenv("MAESTRO_TRADE_PATH", "/api/v1/trade")

    def is_configured(self) -> bool:
        return bool(self.api_base_url and self.api_key and self.account)


class MaestroOrderError(RuntimeError):
    pass


class MaestroClient:
    def __init__(self, config: MaestroConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._inflight_pairs: set[str] = set()
        self._thread_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.config.is_configured()

    def _claim_pair(self, pair_addr: str) -> bool:
        if not pair_addr:
            return False
        with self._thread_lock:
            key = pair_addr.lower()
            if key in self._inflight_pairs:
                return False
            self._inflight_pairs.add(key)
            return True

    def _release_pair(self, pair_addr: str) -> None:
        if not pair_addr:
            return
        with self._thread_lock:
            self._inflight_pairs.discard(pair_addr.lower())

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
                self._session = aiohttp.ClientSession(trust_env=True, timeout=timeout)
            return self._session

    async def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.api_base_url}{self.config.trade_path}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        session = await self._get_session()
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status == 429 and attempt < attempts:
                        delay = self.config.retry_backoff * attempt
                        logger.warning("Maestro rate limited; retrying in %.2fs", delay)
                        await asyncio.sleep(delay)
                        continue
                    if resp.status >= 400:
                        raise MaestroOrderError(
                            f"Maestro API error {resp.status}: {text[:500]}"
                        )
                    try:
                        return await resp.json()
                    except Exception:
                        return {"raw": text}
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if attempt >= attempts:
                    raise MaestroOrderError(f"Maestro request failed: {exc}")
                backoff = self.config.retry_backoff * attempt
                logger.warning("Maestro request error, retrying in %.2fs", backoff)
                await asyncio.sleep(backoff)
        raise MaestroOrderError("Maestro request exhausted retries")

    async def submit_buy(
        self,
        pair_address: str,
        token_out: str,
        amount_eth: Decimal,
        *,
        token_in: str,
        token_symbol: str = "",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        if amount_eth <= 0:
            raise MaestroOrderError("Non-positive buy amount")
        payload = {
            "type": "buy",
            "pairAddress": pair_address,
            "account": self.config.account,
            "tokenIn": token_in,
            "tokenOut": token_out,
            "amountIn": str(amount_eth),
            "amountInCurrency": "ETH",
            "slippageBps": self.config.slippage_bps,
            "dryRun": self.config.dry_run,
            "metadata": {
                "symbol": token_symbol,
                "reason": reason or "passing_pair",
            },
        }
        if self.config.priority_fee_gwei:
            payload["priorityFeeGwei"] = self.config.priority_fee_gwei
        started = time.perf_counter()
        result = await self._request(payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Maestro buy submitted for %s (%.6f ETH) in %sms",
            token_symbol or token_out,
            float(amount_eth),
            elapsed_ms,
        )
        return result

    def submit_buy_background(
        self,
        pair_address: str,
        token_out: str,
        amount_eth: Decimal,
        *,
        token_in: str,
        token_symbol: str = "",
        reason: Optional[str] = None,
        callback: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            logger.debug("Maestro disabled or misconfigured; skipping buy trigger")
            return
        if not self._claim_pair(pair_address):
            logger.debug("Maestro buy already in-flight for %s", pair_address)
            return

        def _runner():
            try:
                asyncio.run(
                    self.submit_buy(
                        pair_address,
                        token_out,
                        amount_eth,
                        token_in=token_in,
                        token_symbol=token_symbol,
                        reason=reason,
                    )
                )
                if callback:
                    callback(success=True, pair=pair_address)
            except Exception as exc:  # pragma: no cover - network path
                logger.error("Maestro buy failed for %s: %s", pair_address, exc)
                if callback:
                    callback(success=False, pair=pair_address, error=str(exc))
            finally:
                self._release_pair(pair_address)

        threading.Thread(target=_runner, daemon=True).start()

