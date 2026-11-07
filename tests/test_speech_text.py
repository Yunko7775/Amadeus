import unittest

from desktop.desktop_app import normalize_japanese_tts_text


class JapaneseTtsTextTests(unittest.TestCase):
    def test_normalizes_kurisu_name_variants_to_their_japanese_reading(self):
        self.assertEqual(
            normalize_japanese_tts_text(
                "牧瀬紅莉栖、牧瀬 红莉栖、牧濑紅莉栖和红莉栖。"
            ),
            "まきせくりす、まきせくりす、まきせくりす和くりす。",
        )

    def test_leaves_unrelated_japanese_text_unchanged(self):
        text = "今日は研究室で実験をする予定です。"

        self.assertEqual(normalize_japanese_tts_text(text), text)


if __name__ == "__main__":
    unittest.main()
