from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from google import genai
from google.genai import types


_RETRYABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]
_Result = TypeVar("_Result")


def call_with_model_fallback(
    call: Callable[..., _Result],
    *,
    model: str,
    fallback_model: str,
    **kwargs: Any,
) -> _Result:
    try:
        return call(model=model, **kwargs)
    except Exception as error:
        if getattr(error, "code", None) != 429 or fallback_model == model:
            raise
        print(
            f"[gemini] {model} returned HTTP 429; "
            f"retrying with {fallback_model}"
        )
        return call(model=fallback_model, **kwargs)


def generate_content_stream_with_fallback(
    models: Any,
    *,
    model: str,
    fallback_model: str,
    **kwargs: Any,
) -> Iterator[Any]:
    try:
        stream = models.generate_content_stream(model=model, **kwargs)
        iterator = iter(stream)
        first_chunk = next(iterator, None)
    except Exception as error:
        if getattr(error, "code", None) != 429 or fallback_model == model:
            raise
        print(
            f"[gemini] {model} returned HTTP 429; "
            f"retrying with {fallback_model}"
        )
        stream = models.generate_content_stream(
            model=fallback_model,
            **kwargs,
        )
        yield from stream
        return

    if first_chunk is not None:
        yield first_chunk
    yield from iterator


def create_gemini_client(
    api_key: str,
    *,
    timeout_ms: int | None = None,
    attempts: int = 4,
):
    retry_options = types.HttpRetryOptions(
        attempts=attempts,
        initial_delay=0.5,
        max_delay=5.0,
        exp_base=2.0,
        jitter=0.2,
        http_status_codes=_RETRYABLE_STATUS_CODES,
    )
    http_options = types.HttpOptions(
        timeout=timeout_ms,
        retry_options=retry_options,
    )
    return genai.Client(api_key=api_key, http_options=http_options)
