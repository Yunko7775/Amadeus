import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop import desktop_app


class GeneratedAudioCleanupTests(unittest.TestCase):
    def test_removes_generated_wav_from_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            audio_path = output_directory / "output_test_0.wav"
            audio_path.write_bytes(b"generated audio")

            with patch.object(desktop_app, "TTS_OUTPUT_DIR", output_directory):
                removed = desktop_app.remove_generated_audio_file(audio_path)

            self.assertTrue(removed)
            self.assertFalse(audio_path.exists())

    def test_does_not_remove_files_outside_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "out"
            output_directory.mkdir()
            unrelated_audio = root / "keep.wav"
            unrelated_audio.write_bytes(b"keep")

            with patch.object(desktop_app, "TTS_OUTPUT_DIR", output_directory):
                removed = desktop_app.remove_generated_audio_file(unrelated_audio)

            self.assertFalse(removed)
            self.assertTrue(unrelated_audio.exists())

    def test_does_not_remove_non_generated_wav_from_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            reference = output_directory / "reference.wav"
            reference.write_bytes(b"keep")

            with patch.object(desktop_app, "TTS_OUTPUT_DIR", output_directory):
                removed = desktop_app.remove_generated_audio_file(reference)

            self.assertFalse(removed)
            self.assertTrue(reference.exists())


if __name__ == "__main__":
    unittest.main()
