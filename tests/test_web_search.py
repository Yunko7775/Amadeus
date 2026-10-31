import unittest
from types import SimpleNamespace
from unittest.mock import patch

from web_search import (
    build_interaction_tools,
    extract_urls,
    get_interaction_sources,
    is_local_time_question,
    search_web,
    should_search_web,
)


class WebSearchTests(unittest.TestCase):
    def test_local_time_questions_do_not_use_web_search(self):
        for question in ("現在幾點？", "现在几点了", "現在是幾點鐘"):
            with self.subTest(question=question):
                self.assertTrue(is_local_time_question(question))
                self.assertFalse(should_search_web(question))

    def test_current_information_and_explicit_search_use_web_search(self):
        self.assertTrue(should_search_web("今天台北天氣如何？"))
        self.assertTrue(should_search_web("幫我搜尋最新的 Gemini 新聞"))
        self.assertTrue(should_search_web("幫我搜尋這個主題"))
        self.assertTrue(should_search_web("現在的日本首相是誰？"))

    def test_regular_chat_and_disabled_search_do_not_search(self):
        self.assertFalse(should_search_web("你覺得這個想法怎麼樣？"))
        self.assertFalse(should_search_web("今天我有點累"))
        self.assertFalse(should_search_web("搜尋最新天氣", enabled=False))

    def test_interaction_tools_route_urls_and_current_search(self):
        self.assertEqual(
            build_interaction_tools("https://example.com/article"),
            [{"type": "url_context"}],
        )
        self.assertEqual(
            build_interaction_tools(
                "幫我搜尋這個網址的最新資訊：https://example.com/article"
            ),
            [{"type": "url_context"}],
        )
        self.assertEqual(
            build_interaction_tools("幫我搜尋最新的 Gemini 新聞"),
            [],
        )
        self.assertEqual(build_interaction_tools("一般聊天"), [])
        self.assertEqual(
            build_interaction_tools("https://example.com/article", enabled=False),
            [],
        )
        self.assertEqual(
            build_interaction_tools(
                "幫我搜尋這個網址的最新資訊：https://example.com/article",
                enabled=False,
            ),
            [],
        )

    @patch("web_search.DDGS")
    def test_external_search_returns_unique_http_sources(self, ddgs_class):
        ddgs_class.return_value.text.return_value = [
            {
                "title": "Gemini news",
                "href": "https://example.com/news",
                "body": "A news snippet",
            },
            {
                "title": "Duplicate result",
                "href": "https://example.com/news",
                "body": "Duplicate",
            },
            {
                "title": "Invalid result",
                "href": "javascript:alert(1)",
                "body": "Invalid",
            },
        ]

        self.assertEqual(
            search_web("latest Gemini news", limit=3),
            [
                {
                    "title": "Gemini news",
                    "url": "https://example.com/news",
                    "snippet": "A news snippet",
                }
            ],
        )
        ddgs_class.assert_called_once_with(timeout=8)
        ddgs_class.return_value.text.assert_called_once_with(
            query="latest Gemini news",
            region="wt-wt",
            safesearch="moderate",
            max_results=3,
            backend="duckduckgo,bing,brave",
        )

    @patch("web_search.DDGS")
    def test_external_search_removes_search_instruction_prefix(self, ddgs_class):
        ddgs_class.return_value.text.return_value = []

        search_web("幫我搜尋 Gemini 最新消息")

        self.assertEqual(
            ddgs_class.return_value.text.call_args.kwargs["query"],
            "Gemini 最新消息",
        )

    def test_extracts_unique_urls_and_trims_sentence_punctuation(self):
        self.assertEqual(
            extract_urls(
                "讀讀 https://example.com/article, 再看 "
                "https://example.com/article。以及 https://example.org/page"
            ),
            ["https://example.com/article", "https://example.org/page"],
        )

    def test_extracts_unique_interaction_url_citations(self):
        citation = SimpleNamespace(
            type="url_citation",
            title="Example article",
            url="https://example.com/article",
        )
        response = SimpleNamespace(
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[
                        SimpleNamespace(
                            type="text",
                            annotations=[citation, citation],
                        )
                    ],
                )
            ]
        )

        self.assertEqual(
            get_interaction_sources(response),
            [
                {
                    "title": "Example article",
                    "url": "https://example.com/article",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
