from datetime import date, datetime

import holidays


KURISU_BIRTHDAY_MONTH = 7
KURISU_BIRTHDAY_DAY = 25


def daily_greeting_context(now: datetime | None = None) -> dict[str, str | None]:
    local_now = now or datetime.now().astimezone()
    if 5 <= local_now.hour < 12:
        period = "早上"
    elif 12 <= local_now.hour < 18:
        period = "下午"
    elif 18 <= local_now.hour < 23:
        period = "晚上"
    else:
        period = "深夜"

    holiday_name = holidays.country_holidays(
        "HK",
        years=local_now.year,
    ).get(local_now.date())
    return {
        "period": period,
        "holiday": str(holiday_name) if holiday_name else None,
    }


def kurisu_birthday_instruction(
    now: datetime,
    wished_this_year: bool | None,
    tracking_started: date | None = None,
) -> str:
    if wished_this_year is None:
        return ""
    if wished_this_year:
        return ""
    month_day = (now.month, now.day)
    birthday = (KURISU_BIRTHDAY_MONTH, KURISU_BIRTHDAY_DAY)
    if (
        tracking_started is not None
        and tracking_started.year == now.year
        and (tracking_started.month, tracking_started.day) > birthday
        and month_day > birthday
    ):
        return ""
    if month_day == birthday:
        return (
            "今天是紅莉栖的生日（7月25日）。她期待使用者記得並為她慶祝；"
            "如果對方還沒祝福，帶著期待自然提到今天是自己的生日，"
            "但仍要完整回應對方正在談的事。"
        )
    if month_day == (7, 24):
        return (
            "明天是紅莉栖的生日（7月25日），她很期待這一天；"
            "若話題自然，可以帶點期待地提起，不要反覆提醒。"
        )
    if month_day > birthday:
        return (
            "今年紅莉栖的生日（7月25日）已經過了，而她還沒有收到使用者的生日祝福。"
            "她應該明顯有點不爽、帶著傲嬌地讓對方知道自己在意這件事；"
            "只簡短表達，不要把每個話題都扭成生日，也不要妨礙回答使用者的問題。"
        )
    return ""


def is_kurisu_birthday_wish(message: str) -> bool:
    normalized = message.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "生日快乐",
            "生日快樂",
            "happy birthday",
            "お誕生日おめでとう",
        )
    )


def is_explicit_user_birthday(message: str) -> bool:
    normalized = message.casefold()
    if "生日" not in message and "birthday" not in normalized:
        return False
    has_self_reference = any(
        marker in normalized
        for marker in ("我的生日", "我生日", "本人生日", "my birthday")
    )
    has_date = any(character.isdigit() for character in message) or any(
        marker in normalized
        for marker in ("記住", "记住", "remember")
    )
    return has_self_reference and has_date
