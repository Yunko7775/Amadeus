# Amadeus

Amadeus is a Windows desktop AI companion that brings a Live2D character into
everyday conversations. It combines conversational AI, voice synthesis, and
persistent memory in one interactive application.

## Features

- Chat with Gemini through a desktop interface.
- Interact with a Live2D avatar with emotion-aware expressions, automatic
  blinking and breathing. Motion plays in response to conversation rather than
  cycling automatically while idle.
- Hear Japanese speech generated locally with GPT-SoVITS.
- Click the mouse's **forward side button (XBUTTON2)** or the microphone icon
  beside Send once to start continuous listening; click it again to stop.
  Local Whisper transcribes each utterance, and about 0.5 seconds of silence
  sends the recognized text to Gemini. Listening pauses while Amadeus speaks
  to avoid the microphone picking up its own voice. Gemini streams Japanese
  sentences to GPT-SoVITS as they are generated; the Chinese reply appears as
  soon as Gemini finishes generating it, without waiting for speech playback
  to finish. The raw microphone audio stays on this PC. The Whisper small model
  is stored under `Resources/models/` inside the project and downloaded there
  on first use. Whisper uses an available CUDA GPU (float16) or falls back to
  CPU (int8). Starting to speak immediately stops current playback. Decoding
  is prompted for Traditional Chinese and Amadeus-related names. Set
  `AMADEUS_PUSH_TO_TALK_KEY` in `.env` to change the global shortcut (for
  example, `f8`).
- Voice timing is written to the Flask backend console after playback, measured
  from the start of recording through the end of generated speech. The current
  voice flow plays each complete Japanese sentence while Gemini continues
  generating the rest of the response.
- Use screen observation as a game companion: while enabled, Amadeus checks
  the desktop every 60 seconds and asks Gemini about meaningful visual changes.
  Unchanged scenes are skipped, and observation stops scheduling and speaking
  when disabled.
- Persist every conversation turn in SQLite, restore recent chat after restart,
  and retrieve older user statements by semantic similarity using Gemini
  embeddings. Retrieved history does not use assistant replies as evidence
  about the user.
- After each 20 new user chat messages, consolidate stable personal facts into
  long-term memory in one background Gemini request. Each fact keeps links to
  the user messages that support it; duplicate facts are updated instead of
  copied, and long-term memories are not automatically pruned.
- Search the web when explicitly requested or when a query clearly needs
  current information. Ordinary chat skips the extra Gemini search-classifier
  request; requests that appear to search personal information still use the
  classifier as a privacy check. Search results are read by Gemini through URL
  Context. If Gemini omits citation annotations, Amadeus keeps its answer and
  shows the DDGS result links separately; it saves research notes only when
  Gemini returns citations. If web search finds no results, Amadeus says it
  could not verify the answer. Source links stay separate from the desktop
  conversation text.
- Log per-stage chat timings to the backend console without logging message
  contents. Each `/ask` request reports memory and conversation retrieval,
  search classification, web search, Gemini generation, translation, history
  saving, and total time, plus rolling medians for the last 20 requests.
  For normal chat, `translation_ms` is the separate translation-service wait
  (now zero); bilingual generation time is included in `gemini_ms`.
- Generate the Traditional Chinese reply and Japanese voice text together in
  one Gemini response for ordinary chat, avoiding a separate translation
  service request. Voice interactions also log transcription, `/ask`, TTS
  generation, playback, and end-to-end times to the desktop console.
- Analyze public Bilibili and YouTube videos using available subtitles and up
  to six sampled frames. If captions are unavailable, Amadeus extracts audio
  for Gemini to understand. Videos must be no longer than 10 minutes and the
  download is limited to 100 MB/120 seconds; downloaded media is temporary,
  while sampled frames, subtitles, and fallback audio are sent to Gemini.
- After two hours without user input, try a proactive message and repeat every
  two idle hours; defer briefly while screen analysis or speech is in progress.
- Check the local calendar at midnight; offer one greeting per day on startup
  or the first chat of a new day, including Hong Kong public holidays. The
  midnight check does not interrupt the user with a greeting.
- Remember Kurisu's July 25 birthday: she looks forward to it and may be
  playfully displeased if the day passes without a birthday wish.
- Search when a question needs factual verification, and use saved follow-up
  questions to continue that research during a later idle conversation.
- Occasionally choose a concrete unanswered question from the character's
  interests, verify it with web sources, and save the findings as a research
  note for later follow-up.
- Use the desktop "主動聊聊" button to test a proactive message without waiting
  for the idle timer.

Screen observation is opt-in. While it is on, screenshots of the desktop are
sent to Gemini for analysis.

Video analysis downloads a temporary copy of a supported public video, then
deletes it after the request. Gemini receives the subtitle text and sampled
frames, or extracted audio when no usable subtitles are available. Private,
login-required, playlist, and over-limit videos are not supported.

## Tech Stack

Python · Tkinter · Live2D · Flask · Gemini API · SQLite · DDGS · GPT-SoVITS

## Architecture

Python modules are grouped by responsibility:

- `desktop/`: Tkinter app, Live2D rendering, voice input, and local speech
  recognition.
- `backend/`: Flask API, conversation memory, web search, and video analysis.
- `shared/`: settings, Gemini client configuration, and helpers shared by both apps.
- `main.py`: starts services as Python modules and launches the desktop app after
  the backend is ready.

The Tkinter desktop app renders the Live2D character and sends chat requests to
a local Flask backend. The backend coordinates Gemini, web search, and SQLite
memory; speech is generated by a local GPT-SoVITS service.

Gemini requests retry transient HTTP errors, including 503, up to three times
after the initial attempt with short backoff. If Gemini remains unavailable,
the chat API returns a clear temporary-unavailable response.

The desktop avatar uses the Cubism 3 Kurisu model from
[Akatosh6DW/Amadeus](https://github.com/Akatosh6DW/Amadeus), attributed there to
ai-poet and the original `amadeus-system-new` project. The model remains
copyrighted by its author and is not covered by this project's software
license; see [the asset source notice](Resources/kurisu_cubism3/ASSET_SOURCE.md).
The phone page continues to use the existing model.

Conversation transcripts are stored locally in the SQLite memory database.
Recent exchanges are included first; older user statements are retrieved as
supporting context, with their original wording and timestamps. Gemini embeddings
index the transcript asynchronously; if indexing or semantic retrieval is
unavailable, the app falls back to local keyword matching. Stable long-term
facts are consolidated from batches of 20 user messages, not judged on every
turn. Each consolidated fact retains its source quotes and is semantically
retrieved with embeddings. Older imported memories without source quotes are
marked as unverified context.

## Demo

[Watch the project demonstration](https://www.bilibili.com/video/BV1yvoFBmEjS/)

## Run (Windows)

Install the GPT-SoVITS runtime, create `.env` from `.env.example`, and add your
`GEMINI_API_KEY`. After setup, double-click `run_amadeus.bat` to start all
services and the desktop app. You can also start it from PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

## Use from a phone

For phone access on a trusted private Wi-Fi, generate a long access code:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(24))"
```

Add it to `.env` as `AMADEUS_WEB_ACCESS_CODE=...`, restart Amadeus, and allow
Python on private networks if Windows Firewall asks. On the PC, run `ipconfig`
to find its IPv4 address; open `http://<PC IPv4>:5000` on a phone connected to
the same Wi-Fi and enter the access code. Without this setting, the server
remains local to the PC. The phone page loads the project's Live2D model and
can read Japanese replies using the phone's built-in speech voice when you tap
the voice button under a reply. This voice differs from GPT-SoVITS; game screen
observation and GPT-SoVITS audio continue to run on the PC. Live2D browser
rendering requires an internet connection to load the pinned
[l2d runtime](https://github.com/hacxy/l2d), which includes the Live2D Cubism
SDK. Use of Cubism is subject to the
[Live2D software license](https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html).

This connection uses HTTP and is intended only for a trusted private network.
Do not use it on public Wi-Fi or expose the port to the internet.
