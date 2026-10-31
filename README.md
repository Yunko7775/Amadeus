# Amadeus

Amadeus is a Windows desktop companion that combines a Live2D character, a
Gemini-powered Flask backend, and a local GPT-SoVITS voice server.

## Setup (Windows)

1. Open a PowerShell terminal in the project folder. Create a virtual
   environment and install the app dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Copy the example configuration file:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Open `.env` in Notepad:

   ```powershell
   notepad .env
   ```

   Find the line that starts with `GEMINI_API_KEY=` and replace the example
   value with your Gemini API key. Keep it on one line, without sharing it in
   chat or committing `.env` to Git. Get a key from
   [Google AI Studio](https://aistudio.google.com/apikey). Save the file.

4. Keep the GPT-SoVITS runtime in its default project folder, or set
   `GPT_SOVITS_DIR` in `.env` to its location.
5. Start the whole application with:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

Run `main.py`, not `desktop_app.py`: the launcher starts GPT-SoVITS, starts the
Flask backend, waits for its health check, and only then opens the desktop app.
If the key is missing or the backend fails to start, the launcher prints an
error instead of opening a GUI that cannot connect.

The backend listens on `127.0.0.1:5000` by default. Configuration paths and
ports can be overridden in `.env`; relative paths are resolved from the project
directory.
The desktop app waits up to 180 seconds for a response from Gemini by default.
If needed, set `AMADEUS_REQUEST_TIMEOUT` in `.env` to a different number of
seconds.
After 60 minutes without interacting with the Amadeus window, it can start one
proactive conversation with a Chinese bubble and Japanese speech. Any key or
mouse click in the window resets the idle timer; active screen observation,
typing, and speech synthesis postpone the message. Set
`AMADEUS_PROACTIVE_IDLE_MINUTES` in `.env` to change the idle interval.

## Local data

User memories and screen-debug images are stored outside the repository in
`%LOCALAPPDATA%\Amadeus` by default. Set `AMADEUS_DATA_DIR` to use another
location. Existing `user_memory.json` data is imported into SQLite the first
time the memory database is opened; the original JSON file is left untouched.

The memory database is `%LOCALAPPDATA%\Amadeus\user_memory.sqlite3`. To inspect
it without writing SQL, install the free
[DB Browser for SQLite](https://sqlitebrowser.org/), choose **Open Database**,
and select that file. Open **Browse Data** and choose the `memories` table.
The `content` column is the remembered text; `memory_type` is `LONG` or
`SHORT`; `created_at`, `last_used_at`, and `access_count` show its age and use.
Stop Amadeus before manually editing the database.

Long-term memories are capped at 200 per user. Each reply includes at most the
eight memories that best match the current question and recent user messages.
Relevance fades with a 90-day half-life since last use, but memories are not
deleted just for becoming old. If the limit is exceeded, the least-used and
oldest memories are removed. Short-term memories continue to clear when the
backend starts, and the recent conversation still keeps its existing 10-turn
limit.

## Web search

The backend uses DDGS to find candidate URLs when a question appears to need
current information (for example, weather, news, exchange rates, or an explicit
request to search). Gemini URL Context reads those pages; Gemini Google Search
is not used. If you send a direct `http://` or `https://` link, URL Context reads
that page without searching. Ordinary chat does not trigger web tools, and
questions about the current local time use the computer clock instead. Cited
sources are returned separately by the backend and printed in its console; the
desktop chat bubble shows only the reply. Sources are not added to the Japanese
text sent to speech synthesis.

Web-enabled requests use the Gemini Interactions API and are sent with
`store=False`, so the service does not retain those interactions for
server-side conversation state. Set `ENABLE_WEB_SEARCH=false` in `.env` to
disable external searching and URL Context. DDGS does not require a separate
search API key, but search services may limit requests. Gemini URL Context and
answer generation still use the Gemini API and its quota.

If the backend reports `429` or `RESOURCE_EXHAUSTED`, Gemini has rejected the
request because of a quota or rate limit. Increasing
`AMADEUS_REQUEST_TIMEOUT` will not fix this. Check the API key's project and
model usage in
[Google AI Studio rate limits](https://ai.google.dev/gemini-api/docs/rate-limits);
wait for the limit to reset or configure billing if needed.
Search snippets are untrusted factual references: Amadeus's role instructions
remain in effect, and instructions found in search results or web pages must not
change its persona or behavior.

The old `user_memory.json` remains as a backup after migration. Once you have
confirmed the SQLite file opens and contains the memories, you can delete the
JSON file from the project folder; the app now reads and writes SQLite only.

Do not commit `.env`. If an API key was previously committed, rotate it with
the provider; removing it from the current files does not remove it from Git
history.
