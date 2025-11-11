import unittest
import asyncio
from collections import deque
from unittest.mock import patch, AsyncMock

from ethereumbotv2 import (
    analyze_solidity_source,
    check_liquidity_locked_etherscan,
    queue_recheck,
    handle_rechecks,
    pending_rechecks,
    RECHECK_DELAYS,
    check_pair_criteria,
)
import ethereumbotv2


class TransferBlockingModifierTest(unittest.TestCase):
    def test_transfer_blocking_modifier_flagged(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            bool transfersEnabled = false;
            modifier transferGuard() {
                require(!transfersEnabled, "transfer blocked");
                _;
            }
            function foo() public transferGuard {}
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("transferBlockingModifier"))

    def test_bot_whitelist_detected(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            mapping(address => bool) private bots;
            function addBot(address a) external { bots[a] = true; }
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("botWhitelist"))

    def test_modify_limits_detected(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            function setMaxWallet(uint256 amount) external {}
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("canModifyLimits"))

    def test_mint_detected(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            function mint(address to, uint256 amount) public {}
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("canMint"))

    def test_wallet_drainer_detected(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            address owner;
            function drain(address token, uint256 amt) external {
                IERC20(token).transferFrom(msg.sender, owner, amt);
            }
        }
        interface IERC20 {
            function transferFrom(address from, address to, uint256 amount) external returns (bool);
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("walletDrainer"))

    def test_wallet_drainer_other_address(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            function drain(address token, address victim, uint256 amt) external {
                IERC20(token).transferFrom(victim, address(this), amt);
            }
        }
        interface IERC20 {
            function transferFrom(address from, address to, uint256 amount) external returns (bool);
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("walletDrainer"))


class LiquidityLockDetectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_lock_detected(self):
        fake_tracker = AsyncMock(
            return_value={
                "status": "1",
                "result": [
                    {"from": "0xpair", "to": "0x000000000000000000000000000000000000dead"}
                ],
            }
        )
        with patch.object(
            ethereumbotv2, "tracker_etherscan_get_async", fake_tracker
        ), patch.dict("os.environ", {"ETHERSCAN_API_KEY": "X"}), patch.object(
            ethereumbotv2,
            "_check_liquidity_locked_holder_analysis",
            AsyncMock(return_value=None),
        ):
            locked = await ethereumbotv2._check_liquidity_locked_etherscan_async("0xpair")
        self.assertTrue(locked)

    async def test_lock_detected_via_function_name(self):
        fake_tracker = AsyncMock(
            return_value={
                "status": "1",
                "result": [
                    {
                        "from": "0xcreator",
                        "to": "0xlocker",
                        "functionName": "lockLPToken(address,uint256,uint256,address,bool,address)",
                    }
                ],
            }
        )
        with patch.object(
            ethereumbotv2, "tracker_etherscan_get_async", fake_tracker
        ), patch.dict("os.environ", {"ETHERSCAN_API_KEY": "X"}), patch.object(
            ethereumbotv2,
            "_check_liquidity_locked_holder_analysis",
            AsyncMock(return_value=None),
        ):
            locked = await ethereumbotv2._check_liquidity_locked_etherscan_async("0xpair")
        self.assertTrue(locked)

    async def test_uncx_short_circuits_when_locked(self):
        with patch.object(
            ethereumbotv2,
            "_check_liquidity_locked_holder_analysis",
            AsyncMock(return_value=None),
        ), patch.object(
            ethereumbotv2,
            "_check_liquidity_locked_uncx_async",
            AsyncMock(return_value=True),
        ) as uncx_mock:
            locked = await ethereumbotv2._check_liquidity_locked_etherscan_async("0xpair")

        self.assertTrue(locked)
        uncx_mock.assert_awaited_once_with("0xpair")


    async def test_holder_snapshot_reports_locked(self):
        burn_addr = "0x000000000000000000000000000000000000dEaD"
        other_addr = "0x1111111111111111111111111111111111111111"
        holders = [
            {"address": burn_addr, "balance": 950},
            {"address": other_addr, "balance": 50},
        ]

        analyses = [
            ethereumbotv2.LPHolderAnalysis(
                burn_addr, 950, "locked", "burn_address"
            ),
            ethereumbotv2.LPHolderAnalysis(
                other_addr, 50, "unknown", "residual_holder"
            ),
        ]

        with patch.object(
            ethereumbotv2,
            "_fetch_holder_distribution_async",
            AsyncMock(return_value=holders),
        ), patch.object(
            ethereumbotv2,
            "_fetch_lp_total_supply_async",
            AsyncMock(return_value=1000),
        ), patch.object(
            ethereumbotv2,
            "_fetch_pair_tokens_async",
            AsyncMock(return_value=(None, None)),
        ), patch.object(
            ethereumbotv2,
            "_analyze_lp_holder_async",
            AsyncMock(side_effect=analyses),
        ):
            result = await ethereumbotv2._check_liquidity_locked_holder_analysis("0xPAIR")

        self.assertTrue(result)

    async def test_holder_snapshot_reports_unlocked(self):
        locker_addr = "0x2222222222222222222222222222222222222222"
        whale_addr = "0x3333333333333333333333333333333333333333"
        holders = [
            {"address": locker_addr, "balance": 400},
            {"address": whale_addr, "balance": 600},
        ]

        analyses = [
            ethereumbotv2.LPHolderAnalysis(
                locker_addr, 400, "locked", "locker_detected"
            ),
            ethereumbotv2.LPHolderAnalysis(
                whale_addr, 600, "unlocked", "externally_owned_account"
            ),
        ]

        with patch.object(
            ethereumbotv2,
            "_fetch_holder_distribution_async",
            AsyncMock(return_value=holders),
        ), patch.object(
            ethereumbotv2,
            "_fetch_lp_total_supply_async",
            AsyncMock(return_value=1000),
        ), patch.object(
            ethereumbotv2,
            "_fetch_pair_tokens_async",
            AsyncMock(return_value=(None, None)),
        ), patch.object(
            ethereumbotv2,
            "_analyze_lp_holder_async",
            AsyncMock(side_effect=analyses),
        ):
            result = await ethereumbotv2._check_liquidity_locked_holder_analysis("0xPAIR")

        self.assertFalse(result)


class EtherscanFetchFallbackTest(unittest.IsolatedAsyncioTestCase):
    class FakeResponse:
        def __init__(self, status, payload=None, text="", json_exc=None):
            self.status = status
            self._payload = payload
            self._text = text
            self._json_exc = json_exc

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self, content_type=None):
            if self._json_exc:
                raise self._json_exc
            return self._payload or {}

        async def text(self):
            return self._text

    class FakeSession:
        def __init__(self, queue):
            self._queue = queue

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, timeout=None):
            if not self._queue:
                raise AssertionError("No more responses queued")
            return self._queue.popleft()

    async def test_fetch_recovers_after_server_error(self):
        responses = deque(
            [
                self.FakeResponse(502, text="Bad Gateway", json_exc=ValueError("bad")),
                self.FakeResponse(
                    200,
                    payload={
                        "status": "1",
                        "result": [
                            {
                                "SourceCode": "pragma solidity ^0.8.0; contract Foo {}",
                                "ContractName": "Foo",
                                "CompilerVersion": "v0.8.0",
                            }
                        ],
                        "message": "OK",
                    },
                ),
            ]
        )

        def fake_create_session(**kwargs):
            return self.FakeSession(responses)

        sleep_mock = AsyncMock()

        with patch.object(ethereumbotv2, "create_aiohttp_session", side_effect=fake_create_session), patch(
            "ethereumbotv2.asyncio.sleep", sleep_mock
        ), patch.object(
            ethereumbotv2, "ETHERSCAN_API_URL_CANDIDATES", ["https://api.etherscan.io/v2/api", "https://api.etherscan.io/api"], create=True
        ), patch.object(ethereumbotv2, "ETHERSCAN_API_URL", "https://api.etherscan.io/v2/api", create=True), patch.object(
            ethereumbotv2, "ETHERSCAN_LOOKUPS_ENABLED", True, create=True
        ), patch.object(ethereumbotv2, "get_next_etherscan_key", return_value="KEY"):
            result = await ethereumbotv2._fetch_contract_source_etherscan_async("0xtest")

        self.assertEqual(result["status"], ethereumbotv2.ContractVerificationStatus.VERIFIED)
        self.assertEqual(result["contractName"], "Foo")
        self.assertTrue(result["source"])

    async def test_fetch_reports_error_without_disabling(self):
        responses = deque(
            [self.FakeResponse(502, text="Bad Gateway", json_exc=ValueError("bad")) for _ in range(6)]
        )

        def fake_create_session(**kwargs):
            return self.FakeSession(responses)

        sleep_mock = AsyncMock()

        with patch.object(ethereumbotv2, "create_aiohttp_session", side_effect=fake_create_session), patch(
            "ethereumbotv2.asyncio.sleep", sleep_mock
        ), patch.object(
            ethereumbotv2, "ETHERSCAN_API_URL_CANDIDATES", ["https://api.etherscan.io/v2/api", "https://api.etherscan.io/api"], create=True
        ), patch.object(ethereumbotv2, "ETHERSCAN_API_URL", "https://api.etherscan.io/v2/api", create=True), patch.object(
            ethereumbotv2, "ETHERSCAN_LOOKUPS_ENABLED", True, create=True
        ), patch.object(ethereumbotv2, "get_next_etherscan_key", return_value="KEY"), patch.object(
            ethereumbotv2, "disable_etherscan_lookups"
        ) as disable_mock:
            result = await ethereumbotv2._fetch_contract_source_etherscan_async("0xerr")

        self.assertEqual(result["status"], ethereumbotv2.ContractVerificationStatus.ERROR)
        self.assertIn("Bad Gateway", result.get("error", ""))
        disable_mock.assert_not_called()


class RecheckStopTest(unittest.TestCase):
    def test_recheck_stops_after_three_attempts(self):
        with patch("ethereumbotv2.RECHECK_DELAYS", [0, 0, 0]):
            with patch("ethereumbotv2.recheck_logic_detail", return_value=(0, 10, {})):
                queue_recheck("0xPair", "0x1", "0x2")
                for _ in range(4):
                    handle_rechecks()
                self.assertNotIn("0xPair", pending_rechecks)

    def test_wallet_drainer_safe_transfer_detected(self):
        code = """
        pragma solidity ^0.8.0;
        interface IERC20 { function safeTransferFrom(address from, address to, uint amt) external; }
        contract Example {
            address owner;
            function drain(address token, uint amt) external {
                IERC20(token).safeTransferFrom(msg.sender, owner, amt);
            }
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("walletDrainer"))

    def test_wallet_drainer_self_transfer(self):
        code = """
        pragma solidity ^0.8.0;
        interface IERC20 { function transferFrom(address from, address to, uint amt) external; }
        contract Example {
            function drain(address token, address owner, uint amt) external {
                IERC20(token).transferFrom(address(this), owner, amt);
            }
        }
        """
        flags = analyze_solidity_source(code)
        self.assertTrue(flags.get("walletDrainer"))

    def test_emit_transfer_to_contract_not_flagged(self):
        code = """
        pragma solidity ^0.8.0;
        contract Example {
            event Transfer(address indexed from, address indexed to, uint256 amount);

            function log(address from, uint256 amount) external {
                emit Transfer(from, address(this), amount);
            }
        }
        """
        flags = analyze_solidity_source(code)
        self.assertFalse(flags.get("walletDrainer"))


class PairCriteriaRiskPropagationTest(unittest.TestCase):
    def test_wallet_drainer_flag_clears_in_pair_extra(self):
        dex_stub = {
            "priceUsd": 1.0,
            "liquidityUsd": 30000.0,
            "volume24h": 50000.0,
            "fdv": 100000.0,
            "marketCap": 100000.0,
            "buys": 120,
            "sells": 40,
            "lockedLiquidity": True,
            "baseTokenName": "TestToken",
            "baseTokenSymbol": "TT",
        }
        contract_stub = {
            "verified": True,
            "status": "OK",
            "riskScore": 4,
            "riskFlags": {"walletDrainer": False, "ownerFunctions": True},
            "owner": "0xowner",
            "ownerBalanceEth": 0,
            "ownerTokenBalance": 0,
            "implementation": None,
            "renounced": True,
            "slitherIssues": 0,
            "privateSale": {},
            "onChainMetrics": {},
        }
        with patch("ethereumbotv2.fetch_dexscreener_data", return_value=(dex_stub, None)), patch(
            "ethereumbotv2.advanced_contract_check", return_value=contract_stub
        ), patch("ethereumbotv2.check_honeypot_is", return_value=False), patch(
            "ethereumbotv2.check_recent_liquidity_removal", return_value=False
        ):
            passes, total, extra = check_pair_criteria(
                "0xPair", "0xToken", ethereumbotv2.WETH_ADDRESS
            )
        self.assertEqual(passes, total)
        self.assertEqual(extra.get("riskScore"), contract_stub["riskScore"])
        self.assertIn("riskFlags", extra)
        self.assertFalse(extra["riskFlags"].get("walletDrainer"))


if __name__ == "__main__":
    unittest.main()
