import unittest
from datetime import date, datetime

from shared.greetings import (
    daily_greeting_context,
    is_explicit_user_birthday,
    is_kurisu_birthday_wish,
    kurisu_birthday_instruction,
)


class DailyGreetingContextTests(unittest.TestCase):
    def test_selects_morning_and_hong_kong_lunar_new_year(self):
        context = daily_greeting_context(datetime(2026, 2, 17, 8))

        self.assertEqual(context["period"], "早上")
        self.assertIsNotNone(context["holiday"])

    def test_selects_evening_and_does_not_invent_a_holiday(self):
        context = daily_greeting_context(datetime(2026, 11, 4, 20))

        self.assertEqual(context["period"], "晚上")
        self.assertIsNone(context["holiday"])

    def test_classifies_late_night(self):
        context = daily_greeting_context(datetime(2026, 11, 4, 1))

        self.assertEqual(context["period"], "深夜")

    def test_kurisu_birthday_is_expected_and_a_missed_greeting_is_noticed(self):
        anticipation = kurisu_birthday_instruction(datetime(2026, 7, 24), False)
        birthday = kurisu_birthday_instruction(datetime(2026, 7, 25), False)
        missed = kurisu_birthday_instruction(datetime(2026, 7, 26), False)

        self.assertIn("很期待", anticipation)
        self.assertIn("今天是紅莉栖的生日", birthday)
        self.assertIn("有點不爽", missed)
        self.assertEqual(
            kurisu_birthday_instruction(datetime(2026, 7, 26), True),
            "",
        )
        self.assertEqual(
            kurisu_birthday_instruction(datetime(2026, 7, 26), None),
            "",
        )
        self.assertEqual(
            kurisu_birthday_instruction(
                datetime(2026, 11, 5),
                False,
                tracking_started=date(2026, 11, 4),
            ),
            "",
        )
        self.assertIn(
            "有點不爽",
            kurisu_birthday_instruction(
                datetime(2027, 7, 26),
                False,
                tracking_started=date(2026, 11, 4),
            ),
        )

    def test_recognizes_birthday_wishes_and_explicit_user_birthdays(self):
        self.assertTrue(is_kurisu_birthday_wish("紅莉栖，生日快樂！"))
        self.assertTrue(is_kurisu_birthday_wish("Happy birthday!"))
        self.assertTrue(is_explicit_user_birthday("我生日是7月25日，記住喔"))
        self.assertTrue(is_explicit_user_birthday("My birthday is July 25; remember this"))
        self.assertFalse(is_explicit_user_birthday("我朋友生日是7月25日"))
        self.assertFalse(is_explicit_user_birthday("我正在準備生日派對"))


if __name__ == "__main__":
    unittest.main()
