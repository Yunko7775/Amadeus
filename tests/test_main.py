import io
import unittest
from unittest.mock import Mock, patch

import main


class MainTests(unittest.TestCase):
    @patch("main.start_console")
    def test_missing_api_key_exits_before_starting_services(self, start_console):
        with patch.object(main, "GEMINI_API_KEY", ""):
            with self.assertRaisesRegex(SystemExit, "Copy .env.example to .env"):
                main.main()

        start_console.assert_not_called()

    @patch("main.urlopen")
    def test_backend_wait_returns_after_healthy_response(self, urlopen):
        response = io.BytesIO(b'{"status": "ok"}')
        response.status = 200
        urlopen.return_value = response
        backend_process = Mock()
        backend_process.poll.return_value = None

        main.wait_for_backend(backend_process, timeout=1)

        urlopen.assert_called_once_with(
            f"{main.BACKEND_BASE_URL}/health",
            timeout=1,
        )


if __name__ == "__main__":
    unittest.main()
