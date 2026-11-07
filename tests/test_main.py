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

    def test_main_launches_backend_and_desktop_as_modules(self):
        backend_process = Mock()
        with patch.object(main, "GEMINI_API_KEY", "test-key"), patch.object(
            main, "GPT_SOVITS_DIR"
        ) as sovits_dir, patch.object(
            main, "PYTHON"
        ) as python, patch.object(
            main, "PYTHONW"
        ) as pythonw, patch.object(
            main, "start_console", side_effect=[Mock(), backend_process]
        ) as start_console, patch.object(
            main, "wait_for_backend"
        ), patch.object(
            main, "start_gui"
        ) as start_gui, patch.object(
            main.time, "sleep"
        ):
            sovits_dir.__truediv__.return_value.is_file.return_value = True
            python.is_file.return_value = True
            pythonw.is_file.return_value = True
            main.main()

        self.assertEqual(
            start_console.call_args_list[1].args[0][1:],
            ["-m", "backend.flask_backend"],
        )
        self.assertEqual(
            start_gui.call_args.args[0][1:],
            ["-m", "desktop.desktop_app"],
        )


if __name__ == "__main__":
    unittest.main()
