import unittest
from unittest.mock import Mock, patch

from desktop import desktop_app


class VoiceInteractionTests(unittest.TestCase):
    def make_app(self):
        app = desktop_app.Live2DChatApp.__new__(desktop_app.Live2DChatApp)
        app.master = Mock()
        app.closing = False
        app.speech_generation = 3
        app.chat_generation = 0
        app.chat_request_active = True
        app.voice_input_active = False
        app.voice_recording_active = False
        app.voice_input_generation = 0
        app.speech_active = True
        app._active_stream_timing = None
        app.activity_generation = 0
        app.live2d = Mock()
        app.user_input = Mock()
        app.show_voice_status = Mock()
        app.display_message = Mock()
        app.send_message = Mock()
        app.btn_voice = Mock()
        return app

    def test_starting_voice_input_invalidates_playback_and_activity(self):
        app = self.make_app()

        app.on_voice_recording_changed(True)

        self.assertEqual(app.speech_generation, 4)
        self.assertEqual(app.chat_generation, 1)
        self.assertFalse(app.chat_request_active)
        self.assertTrue(app.voice_input_active)
        self.assertFalse(app.speech_active)
        app.master.after.call_args_list[0].args[1]()
        app.btn_voice.config.assert_called_once_with(
            text="🔴",
            bg="#f44336",
        )
        app.master.after.call_args_list[1].args[1]()
        app.live2d.stop_audio.assert_called_once_with()

    def test_voice_transcript_starts_the_streaming_reply_flow(self):
        app = self.make_app()
        timing = {"started_at": 1.0}
        app.start_streaming_voice_reply = Mock()

        app.send_voice_transcript(
            "今天的實驗結果出來了",
            voice_generation=0,
            chat_generation=0,
            timing=timing,
        )

        app.user_input.delete.assert_called_once_with(0, "end")
        app.user_input.insert.assert_not_called()
        app.start_streaming_voice_reply.assert_called_once_with(
            "今天的實驗結果出來了",
            timing,
        )
        app.send_message.assert_not_called()

    def test_completed_utterance_starts_local_transcription_before_release(self):
        app = self.make_app()
        app.transcribe_voice_input = Mock()
        audio = object()

        with patch.object(desktop_app.time, "perf_counter", return_value=2.0):
            app.on_voice_utterance_audio(
                audio,
                recording_id=0,
                started_at=1.0,
                ended_at=1.8,
            )

        app.transcribe_voice_input.assert_called_once()
        args = app.transcribe_voice_input.call_args.args
        self.assertIs(args[0], audio)
        self.assertEqual(args[1:3], (0, 0))
        self.assertEqual(args[3]["recording_ms"], 800.0)

    def test_empty_utterance_keeps_listening_without_timing_noise(self):
        app = self.make_app()
        app.voice_recording_active = True

        app.finish_empty_voice_utterance(generation=0)

        self.assertTrue(app.voice_input_active)
        app.show_voice_status.assert_called_once_with(
            "🎙 持續聆聽中：停頓約 0.5 秒開始回覆"
        )

    def test_blank_whisper_result_does_not_log_no_transcript(self):
        app = self.make_app()
        app.voice_recording_active = True
        timing = {"started_at": 1.0}

        with patch.object(desktop_app, "transcribe_chinese", return_value=""):
            with patch.object(
                desktop_app.Live2DChatApp,
                "log_voice_timing",
            ) as log:
                app.transcribe_voice_input(
                    audio=object(),
                    voice_generation=0,
                    chat_generation=0,
                    timing=timing,
                )

        log.assert_not_called()
        app.master.after.assert_called_once()
        scheduled = app.master.after.call_args.args[1]
        scheduled()
        app.show_voice_status.assert_called_once_with(
            "🎙 持續聆聽中：停頓約 0.5 秒開始回覆"
        )

    def test_stale_voice_transcript_is_not_sent(self):
        app = self.make_app()
        app.chat_generation = 2

        app.send_voice_transcript(
            "比較早的語音",
            voice_generation=0,
            chat_generation=1,
        )

        app.user_input.delete.assert_not_called()
        app.send_message.assert_not_called()

    def test_interruption_stops_the_live2d_audio(self):
        app = self.make_app()

        app.interrupt_speech_playback()

        app.live2d.stop_audio.assert_called_once_with()

    def test_microphone_and_transcription_count_as_busy(self):
        app = self.make_app()
        app.user_input.get.return_value = ""
        app.is_screen_processing = False
        app.proactive_request_active = False
        app.chat_request_active = False
        app.voice_input_active = True

        self.assertTrue(app.is_proactive_busy())

        app.voice_input_active = False
        app.chat_request_active = True
        self.assertTrue(app.is_proactive_busy())

    def test_stale_speech_generation_cannot_finish_a_newer_reply(self):
        app = self.make_app()
        app.speech_generation = 5

        app.finish_speech(generation=4)

        self.assertTrue(app.speech_active)

    def test_cancelled_speech_producer_does_not_generate_more_chunks(self):
        app = self.make_app()
        app.speech_generation = 2
        app.generate_single_audio = Mock()
        started_threads = []

        class FakeThread:
            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon
                started_threads.append(self)

            def start(self):
                pass

        with patch.object(desktop_app.threading, "Thread", FakeThread):
            app.process_and_stream(
                "這是測試回覆。",
                "これはテストです。",
            )

        app.speech_generation += 1
        producer = started_threads[0]
        producer.target()

        app.generate_single_audio.assert_not_called()

    def test_stream_reply_is_displayed_before_speech_playback_finishes(self):
        app = self.make_app()
        event = {
            "reply": "這是中文回覆。",
            "expression": "smile1",
        }

        app.display_stream_reply(
            event,
            chat_generation=0,
            speech_generation=3,
        )

        app.live2d.set_expression.assert_called_once_with("smile1")
        app.display_message.assert_called_once_with(
            "這是中文回覆。",
            auto_hide=False,
        )
        self.assertFalse(app.chat_request_active)

    def test_stale_stream_reply_is_not_displayed(self):
        app = self.make_app()
        app.chat_generation = 2

        app.display_stream_reply(
            {"reply": "過期的回覆。"},
            chat_generation=1,
            speech_generation=3,
        )

        app.display_message.assert_not_called()
        app.live2d.set_expression.assert_not_called()

    def test_voice_timing_is_sent_to_the_backend_log(self):
        timing = {
            "started_at": 0.0,
            "transcription_ms": 100.0,
            "ask_request_ms": 500.0,
            "ask_server_ms": {
                "gemini_ms": 200.0,
                "translation_ms": 150.0,
            },
            "tts_generation_ms": 50.0,
            "tts_playback_ms": 300.0,
            "tts_pipeline_ms": 320.0,
        }

        class ImmediateThread:
            def __init__(self, *, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon

            def start(self):
                self.target(*self.args)

        response = Mock()
        with patch.object(
            desktop_app.threading,
            "Thread",
            ImmediateThread,
        ):
            with patch.object(
                desktop_app.requests,
                "post",
                return_value=response,
            ) as post:
                desktop_app.Live2DChatApp.log_voice_timing(
                    timing,
                    "completed",
                )

        post.assert_called_once()
        self.assertTrue(post.call_args.args[0].endswith("/voice_timing"))
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["outcome"], "completed")
        self.assertEqual(payload["timings"]["translation_ms"], 150.0)
        self.assertEqual(payload["timings"]["tts_pipeline_ms"], 320.0)
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
