import re


EXPRESSION_ALIASES = {
    "happy": ("happy", "幸せ", "ハッピー", "満足しています"),
    "angry": ("angry", "怒り"),
    "neutral": ("neutral", "ニュートラル", "ニュートン", "中立"),
    "surprised": ("surprised", "びっくり", "驚いた"),
    "sad": ("sad", "悲しい"),
    "disgust": ("disgust", "嫌悪感"),
}

_ALIAS_TO_EXPRESSION = {
    alias.casefold(): expression
    for expression, aliases in EXPRESSION_ALIASES.items()
    for alias in aliases
}
_EXPRESSION_TAG = re.compile(r"[（(]\s*([^（）()]+?)\s*[)）]")
_SOURCE_SECTION = re.compile(
    r"(?im)^\s*(?:資料來源|资料来源|參考來源|参考来源|sources?|citations?)\s*[:：]\s*"
)


def normalize_expression(value: str) -> str:
    return _ALIAS_TO_EXPRESSION.get(value.strip().casefold(), "neutral")


def extract_expression_and_text(text: str) -> tuple[str, str]:
    expression = "neutral"
    expression_found = False

    def remove_known_tag(match: re.Match) -> str:
        nonlocal expression, expression_found
        alias = match.group(1).strip().casefold()
        matched_expression = _ALIAS_TO_EXPRESSION.get(alias)
        if matched_expression is None:
            return match.group(0)
        if not expression_found:
            expression = matched_expression
            expression_found = True
        return ""

    display_text = _EXPRESSION_TAG.sub(remove_known_tag, text).strip()
    return expression, display_text


def remove_source_section(text: str) -> str:
    return _SOURCE_SECTION.split(text, maxsplit=1)[0].rstrip()
