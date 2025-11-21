import unittest
from unittest.mock import patch

import social_discovery


class SocialDiscoveryExtractionTests(unittest.TestCase):
    def setUp(self):
        social_discovery.SOCIAL_CACHE.clear()

    def test_extract_links_from_entry_collects_known_fields(self):
        entry = {
            "website": "https://example.org",
            "telegram": "https://t.me/exampletoken",
            "twitter": "https://x.com/exampletoken",
            "description": "community at https://t.me/exampletoken",
        }

        links = social_discovery._extract_links_from_entry(entry)

        self.assertIn("https://example.org", links)
        self.assertIn("https://t.me/exampletoken", links)
        self.assertIn("https://x.com/exampletoken", links)

    def test_extract_links_from_text_captures_social_handles(self):
        text = """
        Visit our hub at https://tokenhub.io and chat via https://t.me/tokenchat.
        Follow announcements on https://twitter.com/token and https://x.com/tokenalt.
        """

        links = social_discovery._extract_links_from_text(text)

        self.assertTrue(any("tokenhub.io" in link for link in links))
        self.assertTrue(any("t.me/tokenchat" in link for link in links))
        self.assertTrue(any("twitter.com/token" in link or "x.com/tokenalt" in link for link in links))


class SocialDiscoveryAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        social_discovery.SOCIAL_CACHE.clear()

    async def test_fetch_social_links_merges_all_sources(self):
        async def fake_meta(session, token_addr):
            return ([{"website": "https://meta.example", "telegram": "https://t.me/meta"}], None)

        async def fake_github(session, links):
            return ["https://x.com/fromreadme"]

        async def fake_search(session, token_addr, pair_addr):
            return ["https://search.example"]

        with patch.object(social_discovery, "_fetch_etherscan_metadata", side_effect=fake_meta), patch.object(
            social_discovery, "_fetch_github_socials_from_links", side_effect=fake_github
        ), patch.object(social_discovery, "_perform_web_search", side_effect=fake_search):
            links, reason = await social_discovery.fetch_social_links_async("0xToken", "0xPair")

        self.assertIsNone(reason)
        self.assertIn("https://meta.example", links)
        self.assertIn("https://t.me/meta", links)
        self.assertIn("https://x.com/fromreadme", links)
        self.assertIn("https://search.example", links)

    async def test_fetch_social_links_uses_pair_when_token_missing(self):
        async def fake_meta(session, token_addr):
            self.assertEqual(token_addr, "0xPair")
            return ([{"website": "https://pair.example"}], None)

        with patch.object(social_discovery, "_fetch_etherscan_metadata", side_effect=fake_meta), patch.object(
            social_discovery, "_fetch_github_socials_from_links", return_value=[]
        ), patch.object(social_discovery, "_perform_web_search", return_value=[]):
            links, reason = await social_discovery.fetch_social_links_async("", "0xPair")

        self.assertIsNone(reason)
        self.assertEqual(links, ["https://pair.example"])

    async def test_fetch_social_links_fall_back_to_contract_source(self):
        async def fake_meta(session, token_addr):
            return ([], "not_listed")

        async def fake_contract_source(session, token_addr):
            self.assertEqual(token_addr, "0xToken")
            return (["https://source.example", "https://t.me/source"], None)

        with patch.object(social_discovery, "_fetch_etherscan_metadata", side_effect=fake_meta), patch.object(
            social_discovery, "_fetch_github_socials_from_links", return_value=[]
        ), patch.object(social_discovery, "_perform_web_search", return_value=[]), patch.object(
            social_discovery, "_fetch_contract_source_socials", side_effect=fake_contract_source
        ):
            links, reason = await social_discovery.fetch_social_links_async("0xToken", "")

        self.assertEqual(reason, "not_listed")
        self.assertIn("https://source.example", links)
        self.assertIn("https://t.me/source", links)

    async def test_v2_requests_include_chainid(self):
        captured = {}

        async def fake_fetch_json(session, url, params=None):
            captured["url"] = url
            captured["params"] = params or {}
            return {"status": "0", "message": "NOTOK"}, None

        with patch.object(social_discovery, "ETHERSCAN_BASE_URLS", ["https://api.etherscan.io/v2/api"]), patch.object(
            social_discovery, "_fetch_json", side_effect=fake_fetch_json
        ):
            async with social_discovery._create_session() as session:
                await social_discovery._fetch_etherscan_metadata(session, "0xToken")

        self.assertEqual(captured["params"].get("chainid"), social_discovery.ETHERSCAN_CHAIN_ID)


if __name__ == "__main__":
    unittest.main()
