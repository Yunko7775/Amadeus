import unittest

from expression_manager import (
    extract_expression_and_text,
    normalize_expression,
    remove_source_section,
)


class ExpressionManagerTests(unittest.TestCase):
    def test_normalizes_aliases_case_insensitively(self):
        self.assertEqual(normalize_expression("HAPPY"), "happy")
        self.assertEqual(normalize_expression("びっくり"), "surprised")
        self.assertEqual(normalize_expression("unknown"), "neutral")

    def test_extracts_expression_and_preserves_other_parentheses(self):
        expression, display_text = extract_expression_and_text(
            "（這是備註）你好。（Happy）"
        )

        self.assertEqual(expression, "happy")
        self.assertEqual(display_text, "（這是備註）你好。")

    def test_uses_first_expression_tag(self):
        expression, display_text = extract_expression_and_text(
            "答覆。（neutral）（sad）"
        )

        self.assertEqual(expression, "neutral")
        self.assertEqual(display_text, "答覆。")

    def test_removes_trailing_source_section_from_tk_display_text(self):
        for heading in ("資料來源：", "资料来源:", "Sources:"):
            with self.subTest(heading=heading):
                reply = (
                    f"今天白天晴朗，晚上可能下雨。\n\n{heading}\n"
                    "- 天氣預報: https://example.com/weather"
                )
                self.assertEqual(
                    remove_source_section(reply),
                    "今天白天晴朗，晚上可能下雨。",
                )

    def test_preserves_source_mentions_in_normal_reply(self):
        reply = "這個資料來源很可靠，我參考了氣象署的預報。"

        self.assertEqual(remove_source_section(reply), reply)

if __name__ == "__main__":
    unittest.main()
