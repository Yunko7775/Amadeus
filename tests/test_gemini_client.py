import unittest
from unittest.mock import Mock, patch

from shared.gemini_client import (
    call_with_model_fallback,
    create_gemini_client,
    generate_content_stream_with_fallback,
)


class GeminiClientTests(unittest.TestCase):
    def test_call_retries_with_fallback_model_after_http_429(self):
        quota_error = Exception("quota exhausted")
        quota_error.code = 429
        call = Mock(side_effect=[quota_error, "fallback response"])

        result = call_with_model_fallback(
            call,
            model="gemini-3.5-flash-lite",
            fallback_model="gemini-3.1-flash-lite",
            contents="test",
        )

        self.assertEqual(result, "fallback response")
        self.assertEqual(
            call.call_args_list,
            [
                unittest.mock.call(
                    model="gemini-3.5-flash-lite",
                    contents="test",
                ),
                unittest.mock.call(
                    model="gemini-3.1-flash-lite",
                    contents="test",
                ),
            ],
        )

    def test_call_does_not_retry_non_429_errors(self):
        service_error = Exception("service unavailable")
        service_error.code = 503
        call = Mock(side_effect=service_error)

        with self.assertRaisesRegex(Exception, "service unavailable"):
            call_with_model_fallback(
                call,
                model="gemini-3.5-flash-lite",
                fallback_model="gemini-3.1-flash-lite",
            )

        call.assert_called_once_with(model="gemini-3.5-flash-lite")

    def test_stream_retries_when_quota_error_occurs_before_first_chunk(self):
        quota_error = Exception("quota exhausted")
        quota_error.code = 429

        class FailedStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise quota_error

        models = Mock()
        models.generate_content_stream.side_effect = [
            FailedStream(),
            iter(["fallback chunk"]),
        ]

        chunks = list(
            generate_content_stream_with_fallback(
                models,
                model="gemini-3.5-flash-lite",
                fallback_model="gemini-3.1-flash-lite",
                contents="test",
            )
        )

        self.assertEqual(chunks, ["fallback chunk"])
        self.assertEqual(
            [call.kwargs["model"] for call in models.generate_content_stream.call_args_list],
            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        )

    def test_client_retries_transient_service_errors_with_bounded_backoff(self):
        created_client = Mock()
        with patch(
            "shared.gemini_client.genai.Client",
            return_value=created_client,
        ) as create_client:
            result = create_gemini_client(
                "test-key",
                timeout_ms=30_000,
                attempts=4,
            )

        self.assertIs(result, created_client)
        options = create_client.call_args.kwargs["http_options"]
        retry_options = options.retry_options
        self.assertEqual(options.timeout, 30_000)
        self.assertEqual(retry_options.attempts, 4)
        self.assertEqual(retry_options.initial_delay, 0.5)
        self.assertEqual(retry_options.max_delay, 5.0)
        self.assertEqual(
            retry_options.http_status_codes,
            [408, 429, 500, 502, 503, 504],
        )


if __name__ == "__main__":
    unittest.main()
