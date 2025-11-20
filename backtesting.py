"""
Backtesting utilities for evaluating pairs that cleared the trading checklist.

This module focuses on two responsibilities:

1. Persisting rich snapshots for every pair that reaches "passing" status so we
   can reconstruct the exact checklist context later.
2. Replaying those snapshots against DexScreener's historical candles to
   compute PnL multiples, drawdowns, and other profitability metrics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEXSCREENER_TV_URL = "https://io.dexscreener.com/dex/tradingview/history"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "backtests"
SNAPSHOT_FILENAME = "passing_snapshots.jsonl"


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PassingSnapshot:
    pair_address: str
    token_address: str
    token_symbol: Optional[str]
    token_name: Optional[str]
    timestamp: float
    passes: int
    total_checks: int
    price_usd: Optional[float]
    market_cap: Optional[float]
    liquidity_usd: Optional[float]
    fdv: Optional[float]
    buys: Optional[int]
    sells: Optional[int]
    locked_liquidity: Optional[bool]
    contract_renounced: Optional[bool]
    risk_score: Optional[float]
    check_breakdown: Dict[str, bool]
    context: str
    checklist_payload: Dict[str, object]


@dataclass
class BacktestResult:
    pair_address: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    pnl_multiple: float
    holding_minutes: int
    max_drawdown_pct: float
    peak_multiple: float
    bars: int
    take_profit_hit: bool
    stop_loss_hit: bool


class BacktestEngine:
    """Persist snapshots and replay them using DexScreener candles."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_file = self.data_dir / SNAPSHOT_FILENAME

    def record_pass_snapshot(
        self,
        pair_address: str,
        extra_stats: Dict[str, object],
        passes: int,
        total: int,
        *,
        context: str,
        token0: str,
        token1: str,
    ) -> None:
        """Persist a JSONL record describing a passing pair.

        The snapshot intentionally captures the raw checklist payload so later
        backtests can correlate PnL with specific gate outcomes (e.g. unrenounced
        but otherwise clean vs. fully renounced).
        """

        token_addr = str(extra_stats.get("tokenAddress") or token0 or token1)
        token_symbol = extra_stats.get("tokenSymbol")
        token_name = extra_stats.get("tokenName")

        snapshot = PassingSnapshot(
            pair_address=pair_address,
            token_address=token_addr,
            token_symbol=token_symbol,
            token_name=token_name,
            timestamp=time.time(),
            passes=passes,
            total_checks=total,
            price_usd=_safe_float(extra_stats.get("priceUsd")),
            market_cap=_safe_float(extra_stats.get("marketCap")),
            liquidity_usd=_safe_float(extra_stats.get("liquidityUsd")),
            fdv=_safe_float(extra_stats.get("fdv")),
            buys=_safe_int(extra_stats.get("buys")),
            sells=_safe_int(extra_stats.get("sells")),
            locked_liquidity=_safe_bool(extra_stats.get("lockedLiquidity")),
            contract_renounced=_safe_bool(extra_stats.get("contractRenounced")),
            risk_score=_safe_float(extra_stats.get("riskScore")),
            check_breakdown=dict(extra_stats.get("checkBreakdown") or {}),
            context=context,
            checklist_payload=extra_stats,
        )
        line = json.dumps(asdict(snapshot), default=str)
        try:
            with self.snapshot_file.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
        except OSError:
            logger.exception("Failed to persist backtest snapshot for %s", pair_address)

    def load_snapshots(self) -> List[PassingSnapshot]:
        records: List[PassingSnapshot] = []
        if not self.snapshot_file.exists():
            return records
        with self.snapshot_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                try:
                    data = json.loads(line)
                    records.append(PassingSnapshot(**data))
                except Exception:
                    logger.debug("Skipping malformed snapshot line")
        return records

    async def fetch_candles(
        self,
        pair_address: str,
        *,
        chain: str = "ETHEREUM",
        resolution: str = "15",
        lookback_hours: int = 72,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> List[Candle]:
        """Return TradingView-style candles from DexScreener."""

        end_ts = int(time.time())
        start_ts = end_ts - (lookback_hours * 3600)
        params = {
            "symbol": f"{chain.upper()}:{pair_address}",
            "resolution": resolution,
            "from": start_ts,
            "to": end_ts,
        }
        close_session = False
        if session is None:
            session = aiohttp.ClientSession(trust_env=True)
            close_session = True
        try:
            async with session.get(
                DEXSCREENER_TV_URL, params=params, timeout=30
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        finally:
            if close_session:
                await session.close()

        if str(data.get("s", "")).lower() != "ok":
            raise RuntimeError(f"DexScreener returned status {data.get('s')} for {pair_address}")

        timestamps = data.get("t") or []
        opens = data.get("o") or []
        highs = data.get("h") or []
        lows = data.get("l") or []
        closes = data.get("c") or []
        volumes = data.get("v") or []
        candles: List[Candle] = []
        for idx, ts_val in enumerate(timestamps):
            try:
                candle = Candle(
                    timestamp=int(ts_val),
                    open=float(opens[idx]),
                    high=float(highs[idx]),
                    low=float(lows[idx]),
                    close=float(closes[idx]),
                    volume=float(volumes[idx]) if idx < len(volumes) else 0.0,
                )
                candles.append(candle)
            except (ValueError, TypeError, IndexError):
                continue
        return candles

    async def backtest_snapshot(
        self,
        snapshot: PassingSnapshot,
        *,
        horizon_minutes: int = 240,
        resolution: str = "15",
        chain: str = "ETHEREUM",
        take_profit_multiple: Optional[float] = None,
        stop_loss_multiple: Optional[float] = None,
    ) -> Optional[BacktestResult]:
        """Replay a single snapshot against historical candles."""

        try:
            candles = await self.fetch_candles(
                snapshot.pair_address,
                chain=chain,
                resolution=resolution,
                lookback_hours=max(72, int(horizon_minutes / 60) + 6),
            )
        except aiohttp.ClientResponseError as exc:
            logger.warning(
                "DexScreener rejected %s with status %s: %s",
                snapshot.pair_address,
                exc.status,
                exc.message,
            )
            return None
        except Exception:
            logger.exception("Failed to fetch candles for %s", snapshot.pair_address)
            return None
        if not candles:
            return None
        entry_ts = int(snapshot.timestamp)
        exit_ts = entry_ts + (horizon_minutes * 60)
        relevant = [c for c in candles if c.timestamp >= entry_ts]
        if not relevant:
            relevant = candles[-1:]
        entry_candle = relevant[0]
        exit_candidates = [c for c in relevant if c.timestamp <= exit_ts]
        if exit_candidates:
            exit_candle = exit_candidates[-1]
        else:
            exit_candle = relevant[-1]

        entry_price = entry_candle.close
        exit_price = exit_candle.close
        high_water = entry_price
        low_water = entry_price
        take_profit_hit = False
        stop_loss_hit = False

        for cndl in relevant:
            high_water = max(high_water, cndl.high)
            low_water = min(low_water, cndl.low)
            if take_profit_multiple and not take_profit_hit:
                if cndl.high >= entry_price * take_profit_multiple:
                    take_profit_hit = True
                    exit_price = max(exit_price, entry_price * take_profit_multiple)
                    exit_ts = cndl.timestamp
                    break
            if stop_loss_multiple and not stop_loss_hit:
                if cndl.low <= entry_price * stop_loss_multiple:
                    stop_loss_hit = True
                    exit_price = min(exit_price, entry_price * stop_loss_multiple)
                    exit_ts = cndl.timestamp
                    break

        pnl_multiple = exit_price / entry_price if entry_price else 0.0
        drawdown_pct = ((low_water - entry_price) / entry_price) * 100 if entry_price else 0.0
        peak_multiple = high_water / entry_price if entry_price else 0.0

        return BacktestResult(
            pair_address=snapshot.pair_address,
            entry_time=entry_ts,
            exit_time=exit_ts,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_multiple=pnl_multiple,
            holding_minutes=horizon_minutes,
            max_drawdown_pct=drawdown_pct,
            peak_multiple=peak_multiple,
            bars=len(relevant),
            take_profit_hit=take_profit_hit,
            stop_loss_hit=stop_loss_hit,
        )

    async def backtest_batch(
        self,
        snapshots: Iterable[PassingSnapshot],
        *,
        limit: Optional[int] = None,
        **kwargs,
    ) -> List[BacktestResult]:
        results: List[BacktestResult] = []
        selected = list(snapshots)
        if limit:
            selected = selected[-limit:]
        tasks = [self.backtest_snapshot(s, **kwargs) for s in selected]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if res:
                results.append(res)
        return results


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "true", "yes", "y"}:
            return True
        if cleaned in {"0", "false", "no", "n"}:
            return False
    return None


def _format_result(result: BacktestResult) -> str:
    return (
        f"{result.pair_address} | pnl x{result.pnl_multiple:.2f} | "
        f"peak x{result.peak_multiple:.2f} | drawdown {result.max_drawdown_pct:.2f}% | "
        f"{result.bars} bars"
    )


def _run_cli(args: argparse.Namespace) -> None:
    engine = BacktestEngine()
    snapshots = engine.load_snapshots()
    if args.pair:
        snapshots = [s for s in snapshots if s.pair_address.lower() == args.pair.lower()]
    if not snapshots:
        print("No matching snapshots found.")
        return

    async def _run():
        res = await engine.backtest_batch(
            snapshots,
            limit=args.limit,
            horizon_minutes=args.horizon,
            resolution=args.resolution,
            chain=args.chain,
            take_profit_multiple=args.take_profit,
            stop_loss_multiple=args.stop_loss,
        )
        for r in sorted(res, key=lambda r: r.entry_time):
            print(_format_result(r))

    asyncio.run(_run())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run backtests against passing pairs")
    parser.add_argument("--pair", help="Pair address to backtest", default=None)
    parser.add_argument("--limit", type=int, help="Limit number of snapshots", default=None)
    parser.add_argument("--horizon", type=int, help="Holding period in minutes", default=240)
    parser.add_argument(
        "--resolution", type=str, help="DexScreener TradingView resolution", default="15"
    )
    parser.add_argument("--chain", type=str, help="Chain (ETHEREUM, BSC, etc)", default="ETHEREUM")
    parser.add_argument("--take-profit", type=float, help="Optional take profit multiple", default=None)
    parser.add_argument("--stop-loss", type=float, help="Optional stop loss multiple", default=None)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_cli(_build_arg_parser().parse_args())
