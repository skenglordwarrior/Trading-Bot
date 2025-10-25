import unittest
import asyncio
from unittest.mock import patch, AsyncMock

from TelegramBot import (
    analyze_solidity_source,
    check_liquidity_locked_etherscan,
    queue_recheck,
    handle_rechecks,
    pending_rechecks,
    RECHECK_DELAYS,
)
import TelegramBot


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
        fake_resp = AsyncMock()
        fake_resp.__aenter__.return_value = fake_resp
        fake_resp.json.return_value = {
            "status": "1",
            "result": [
                {"from": "0xpair", "to": "0x000000000000000000000000000000000000dead"}
            ],
        }
        fake_session = AsyncMock()
        def fake_get(*args, **kwargs):
            return fake_resp
        fake_session.get = fake_get
        fake_session.__aenter__.return_value = fake_session
        with patch("aiohttp.ClientSession", return_value=fake_session), patch.dict("os.environ", {"ETHERSCAN_API_KEY": "X"}):
            locked = await TelegramBot._check_liquidity_locked_etherscan_async("0xpair")
        self.assertTrue(locked)


class RecheckStopTest(unittest.TestCase):
    def test_recheck_stops_after_three_attempts(self):
        with patch("TelegramBot.RECHECK_DELAYS", [0, 0, 0]):
            with patch("TelegramBot.recheck_logic_detail", return_value=(0, 10, {})):
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


if __name__ == "__main__":
    unittest.main()
