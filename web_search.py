import re
from typing import Any
from urllib.parse import urlsplit

from ddgs import DDGS


_URL = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]+",
    re.IGNORECASE,
)
_URL_TRAILING_PUNCTUATION = ".,!?;:，。！？、）)]}』」"
_EXPLICIT_SEARCH = re.compile(
    r"搜尋|搜索|查一下|查一查|幫我查|帮我查|上網查|上网查|網路查|网络查|"
    r"google一下|網路搜尋|网络搜索",
    re.IGNORECASE,
)
_SEARCH_PREFIX = re.compile(
    r"^\s*(?:(?:請|请)?(?:幫我|帮我)?(?:上網|上网|網路|网络)?"
    r"(?:搜尋|搜索|查詢|查询|查一下|查一查)|google一下)\s*"
)
_LOCAL_TIME_QUESTION = re.compile(
    r"(?:現在|现在|目前|此刻|當前|当前)?\s*(?:是)?\s*"
    r"(?:幾點|几点|幾點鐘|几点钟|什麼時間|什么时间|幾點了|几点了)",
    re.IGNORECASE,
)
_FRESHNESS_KEYWORDS = (
    "最新",
    "新聞",
    "新闻",
    "天氣",
    "天气",
    "股價",
    "股价",
    "匯率",
    "汇率",
    "比分",
    "賽程",
    "赛程",
    "營業時間",
    "营业时间",
    "開放時間",
    "开放时间",
    "售價",
    "售价",
    "版本",
)
_TEMPORAL_KEYWORDS = ("現在", "现在", "目前", "當前", "当前", "今天", "今日", "本週", "本周", "本月", "今年")
_CURRENT_ENTITY_KEYWORDS = (
    "總統",
    "总统",
    "首相",
    "市長",
    "市长",
    "總理",
    "总理",
    "董事長",
    "董事长",
    "執行長",
    "执行长",
)


def is_local_time_question(question: str) -> bool:
    return bool(_LOCAL_TIME_QUESTION.search(question))


def should_search_web(question: str, enabled: bool = True) -> bool:
    if not enabled or is_local_time_question(question):
        return False
    if _EXPLICIT_SEARCH.search(question):
        return True
    if any(keyword in question for keyword in _FRESHNESS_KEYWORDS):
        return True
    return any(keyword in question for keyword in _TEMPORAL_KEYWORDS) and any(
        keyword in question for keyword in _CURRENT_ENTITY_KEYWORDS
    )


def extract_urls(text: str, limit: int = 20) -> list[str]:
    urls: list[str] = []
    seen_urls: set[str] = set()

    for match in _URL.finditer(text):
        url = match.group(0).rstrip(_URL_TRAILING_PUNCTUATION)
        if url not in seen_urls:
            seen_urls.add(url)
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def search_web(query: str, limit: int = 3) -> list[dict[str, str]]:
    query = _SEARCH_PREFIX.sub("", query, count=1).strip()
    results = DDGS(timeout=8).text(
        query=query,
        region="wt-wt",
        safesearch="moderate",
        max_results=limit,
        backend="duckduckgo,bing,brave",
    )
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result.get("href", "").strip()
        parsed_url = urlsplit(url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or url in seen_urls
        ):
            continue
        seen_urls.add(url)
        sources.append(
            {
                "title": result.get("title", "").strip() or url,
                "url": url,
                "snippet": result.get("body", "").strip(),
            }
        )
        if len(sources) >= limit:
            break
    return sources


def build_interaction_tools(question: str, enabled: bool = True) -> list[dict[str, str]]:
    if not enabled or not extract_urls(question):
        return []

    return [{"type": "url_context"}]


def get_interaction_sources(response: Any, limit: int = 3) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for step in getattr(response, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for content_block in getattr(step, "content", None) or []:
            if getattr(content_block, "type", None) != "text":
                continue
            for annotation in getattr(content_block, "annotations", None) or []:
                url = getattr(annotation, "url", None)
                if (
                    getattr(annotation, "type", None) != "url_citation"
                    or not url
                    or url in seen_urls
                ):
                    continue
                seen_urls.add(url)
                sources.append(
                    {
                        "title": getattr(annotation, "title", None) or url,
                        "url": url,
                    }
                )
                if len(sources) >= limit:
                    return sources
    return sources
