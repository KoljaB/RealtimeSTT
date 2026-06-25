"""Private YouTube/RealtimeSTT synchronization GUI.

The harness downloads one local media file, plays that exact file in a browser
UI, and feeds audio decoded from the same file to RealtimeSTT after the browser
confirms that playback is actually running.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealtimeSTT YouTube Sync</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #06080d;
      --panel: #101622;
      --panel-2: #151d2b;
      --line: #263244;
      --text: #f5f8ff;
      --muted: #8e9bb0;
      --green: #37f59a;
      --cyan: #35d7ff;
      --blue: #4da3ff;
      --yellow: #ffe15c;
      --red: #ff5b7a;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(280px, 42vh) minmax(340px, 1fr);
      gap: 14px;
      padding: 18px;
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }

    input, button {
      border: 1px solid var(--line);
      border-radius: 7px;
      font: inherit;
      min-height: 44px;
    }

    input {
      width: 100%;
      background: #0b1019;
      color: var(--text);
      padding: 0 14px;
      outline: none;
    }

    input:focus {
      border-color: var(--cyan);
      box-shadow: 0 0 0 2px rgba(53, 215, 255, 0.16);
    }

    button {
      background: var(--green);
      color: #03120a;
      padding: 0 18px;
      font-weight: 400;
      cursor: pointer;
    }

    button.secondary {
      background: #1c2637;
      color: var(--text);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .main {
      display: grid;
      grid-template-columns: minmax(420px, 1fr) minmax(260px, 340px);
      gap: 14px;
      min-height: 0;
    }

    .video {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      min-height: 320px;
      position: relative;
    }

    #videoPlayer {
      width: 100%;
      height: 100%;
      min-height: 320px;
      background: #000;
      display: block;
    }

    .placeholder {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 18px;
      pointer-events: none;
    }

    .placeholder.hidden {
      display: none;
    }

    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }

    .status h2, .transcript h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--cyan);
      font-size: 14px;
      font-weight: 400;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    #statusLog {
      margin: 0;
      padding: 12px 14px;
      color: #d5e0f5;
      overflow: auto;
      white-space: pre-wrap;
      font: 13px/1.45 Consolas, "Cascadia Mono", monospace;
    }

    .transcript {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }

    #transcript {
      padding: 18px;
      overflow: auto;
      font-size: 24px;
      line-height: 1.35;
      white-space: pre-wrap;
    }

    .final-a {
      color: var(--blue);
      font-weight: 400;
    }

    .final-b {
      color: var(--yellow);
      font-weight: 400;
    }

    .stable {
      color: #ffffff;
      font-weight: 400;
    }

    .realtime {
      color: #8a93a5;
      font-weight: 400;
    }

    .error {
      color: var(--red);
    }

    @media (max-width: 900px) {
      .app {
        grid-template-rows: auto minmax(420px, auto) minmax(340px, 1fr);
      }

      .toolbar, .main {
        grid-template-columns: 1fr;
      }

      #transcript {
        font-size: 21px;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <input id="urlInput" type="url" autocomplete="off" spellcheck="false" placeholder="https://www.youtube.com/watch?v=...">
      <button id="startButton">Start</button>
      <button id="stopButton" class="secondary">Stop</button>
    </div>

    <main class="main">
      <section class="video">
        <video id="videoPlayer" controls playsinline></video>
        <div id="placeholder" class="placeholder">Downloaded video appears here</div>
      </section>

      <section class="status">
        <h2>Status</h2>
        <pre id="statusLog"></pre>
      </section>
    </main>

    <section class="transcript">
      <h2>Detected Text</h2>
      <div id="transcript"></div>
    </section>
  </div>

  <script>
    const urlInput = document.getElementById("urlInput");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const video = document.getElementById("videoPlayer");
    const placeholder = document.getElementById("placeholder");
    const statusLog = document.getElementById("statusLog");
    const transcript = document.getElementById("transcript");

    let activeSessionId = null;
    let timePing = null;
    let finalTexts = [];
    let stableText = "";
    let realtimeText = "";
    let lastVisibleTranscriptHtml = "";
    let sawStructuredRealtime = false;
    let postedFirstPlaying = false;

    function appendStatus(message, cssClass) {
      const line = document.createElement("div");
      if (cssClass) {
        line.className = cssClass;
      }
      const timestamp = new Date().toLocaleTimeString();
      line.textContent = `[${timestamp}] ${message}`;
      statusLog.appendChild(line);
      statusLog.scrollTop = statusLog.scrollHeight;
    }

    function resetDisplay() {
      statusLog.textContent = "";
      finalTexts = [];
      stableText = "";
      realtimeText = "";
      lastVisibleTranscriptHtml = "";
      sawStructuredRealtime = false;
      postedFirstPlaying = false;
      stopTimePings();
      video.removeAttribute("src");
      video.load();
      placeholder.classList.remove("hidden");
      renderTranscript();
    }

    function renderTranscript() {
      transcript.textContent = "";
      finalTexts.forEach((text, index) => {
        const span = document.createElement("span");
        span.className = index % 2 === 0 ? "final-a" : "final-b";
        span.textContent = text.trim() + " ";
        transcript.appendChild(span);
      });

      if (stableText) {
        const stable = document.createElement("span");
        stable.className = "stable";
        stable.textContent = stableText;
        transcript.appendChild(stable);
      }

      if (realtimeText) {
        const realtime = document.createElement("span");
        realtime.className = "realtime";
        realtime.textContent = realtimeText;
        transcript.appendChild(realtime);
      }

      if (transcript.textContent.trim()) {
        lastVisibleTranscriptHtml = transcript.innerHTML;
      } else if (lastVisibleTranscriptHtml) {
        transcript.innerHTML = lastVisibleTranscriptHtml;
      }

      transcript.scrollTop = transcript.scrollHeight;
    }

    function normalizeForOverlap(text) {
      return (text || "")
        .toLowerCase()
        .replace(/[^\p{L}\p{N}]+/gu, " ")
        .trim()
        .replace(/\s+/g, " ");
    }

    function tokenSpans(text) {
      const spans = [];
      const regex = /[\p{L}\p{N}]+/gu;
      let match;
      while ((match = regex.exec(text || "")) !== null) {
        spans.push({
          token: match[0].toLowerCase(),
          start: match.index,
          end: match.index + match[0].length
        });
      }
      return spans;
    }

    function overlapScore(stableTokens, realtimeTokens, start, length) {
      if (length <= 0) {
        return 0;
      }
      let matches = 0;
      for (let index = 0; index < length; index += 1) {
        if (stableTokens[index] === realtimeTokens[start + index]) {
          matches += 1;
        }
      }
      return matches / length;
    }

    function bestContainedTokenScore(needleTokens, haystackTokens) {
      if (!needleTokens.length || !haystackTokens.length) {
        return 0;
      }

      let best = 0;
      for (let start = 0; start < haystackTokens.length; start += 1) {
        const length = Math.min(needleTokens.length, haystackTokens.length - start);
        if (length <= 0) {
          continue;
        }
        const score = overlapScore(needleTokens, haystackTokens, start, length);
        const coverage = length / Math.max(1, needleTokens.length);
        best = Math.max(best, score * coverage);
      }
      return best;
    }

    function realtimeSuffixAfterStable(displayText, stable) {
      displayText = displayText || "";
      stable = stable || "";
      if (!displayText || !stable) {
        return displayText;
      }

      const foldedDisplay = displayText.toLowerCase();
      const foldedStable = stable.toLowerCase();
      if (foldedDisplay.startsWith(foldedStable)) {
        return displayText.slice(stable.length).replace(/^\s+/, "");
      }

      const stableNorm = normalizeForOverlap(stable);
      const displayNorm = normalizeForOverlap(displayText);
      if (stableNorm && displayNorm.startsWith(stableNorm)) {
        const stableTokenCount = stableNorm.split(" ").length;
        const spans = tokenSpans(displayText);
        if (spans.length >= stableTokenCount) {
          return displayText.slice(spans[stableTokenCount - 1].end).replace(/^\s+/, "");
        }
      }

      const stableSpans = tokenSpans(stable);
      const realtimeSpans = tokenSpans(displayText);
      const stableTokens = stableSpans.map((span) => span.token);
      const realtimeTokens = realtimeSpans.map((span) => span.token);
      if (!stableTokens.length || !realtimeTokens.length) {
        return displayText;
      }

      let best = {score: 0, endToken: 0};
      const minLength = Math.min(stableTokens.length, realtimeTokens.length);
      for (let stableStart = Math.max(0, stableTokens.length - 12); stableStart < stableTokens.length; stableStart += 1) {
        const stableTail = stableTokens.slice(stableStart);
        const maxRealtimeStart = Math.min(8, realtimeTokens.length - 1);
        for (let realtimeStart = 0; realtimeStart <= maxRealtimeStart; realtimeStart += 1) {
          const length = Math.min(stableTail.length, realtimeTokens.length - realtimeStart);
          if (length <= 0) {
            continue;
          }
          const score = overlapScore(stableTail, realtimeTokens, realtimeStart, length);
          const coverage = length / Math.max(1, stableTail.length);
          const weighted = score * coverage;
          if (weighted > best.score && score >= 0.58) {
            best = {score: weighted, endToken: realtimeStart + length};
          }
        }
      }

      if (best.endToken > 0 && best.endToken < realtimeSpans.length) {
        return displayText.slice(realtimeSpans[best.endToken - 1].end).replace(/^\s+/, "");
      }
      if (best.endToken >= realtimeSpans.length && best.score >= 0.72) {
        return "";
      }

      return displayText;
    }

    function liveRemainderAfterFinal(finalText) {
      const live = `${stableText || ""}${realtimeText || ""}`.trim();
      if (!live || !finalText) {
        return "";
      }

      const remainder = realtimeSuffixAfterStable(live, finalText).trim();
      if (remainder && remainder !== live) {
        return remainder;
      }

      const liveNorm = normalizeForOverlap(live);
      const finalNorm = normalizeForOverlap(finalText);
      if (!liveNorm || !finalNorm) {
        return live;
      }
      if (finalNorm.includes(liveNorm)) {
        return "";
      }
      return live;
    }

    function shouldKeepCurrentLive(nextStable, nextRealtime) {
      const current = `${stableText || ""}${realtimeText || ""}`.trim();
      const next = `${nextStable || ""}${nextRealtime || ""}`.trim();
      if (!current || !next) {
        return false;
      }
      if (next.length >= current.length - 8) {
        return false;
      }

      const currentNorm = normalizeForOverlap(current);
      const nextNorm = normalizeForOverlap(next);
      if (!currentNorm || !nextNorm) {
        return false;
      }
      if (currentNorm.startsWith(nextNorm) || currentNorm.includes(nextNorm)) {
        return true;
      }

      const currentTokens = tokenSpans(current).map((span) => span.token);
      const nextTokens = tokenSpans(next).map((span) => span.token);
      if (nextTokens.length + 2 >= currentTokens.length) {
        return false;
      }
      if (bestContainedTokenScore(nextTokens, currentTokens) >= 0.58) {
        return true;
      }
      const length = Math.min(nextTokens.length, currentTokens.length);
      return overlapScore(nextTokens, currentTokens, 0, length) >= 0.62;
    }

    function livePartsAfterFinal(finalText) {
      const liveRemainder = liveRemainderAfterFinal(finalText);
      if (!liveRemainder) {
        return {stable: "", realtime: ""};
      }

      let stableRemainder = "";
      if (stableText) {
        const stableCandidate = realtimeSuffixAfterStable(stableText, finalText).trim();
        if (stableCandidate && stableCandidate !== stableText.trim()) {
          stableRemainder = stableCandidate;
        } else {
          const stableNorm = normalizeForOverlap(stableText);
          const finalNorm = normalizeForOverlap(finalText);
          if (stableNorm && finalNorm && !finalNorm.includes(stableNorm)) {
            stableRemainder = stableText.trim();
          }
        }
      }

      if (stableRemainder) {
        return {
          stable: stableRemainder,
          realtime: realtimeSuffixAfterStable(liveRemainder, stableRemainder).trim()
        };
      }
      return {stable: "", realtime: liveRemainder};
    }

    function splitRealtime(displayText, stable, unstable) {
      displayText = displayText || "";
      stable = stable || "";
      unstable = unstable || "";
      const hasStructuredParts = Boolean(stable || unstable);

      if (!displayText && !stable && !unstable) {
        return;
      }

      if (!hasStructuredParts && sawStructuredRealtime) {
        return;
      }

      let nextStable = stableText;
      let nextRealtime = realtimeText;
      if (hasStructuredParts) {
        sawStructuredRealtime = true;
        nextStable = stable;
        nextRealtime = unstable || realtimeSuffixAfterStable(displayText, stable);
      } else if (stableText && displayText.toLowerCase().startsWith(stableText.toLowerCase())) {
        nextRealtime = displayText.slice(stableText.length);
      } else if (stableText) {
        nextRealtime = realtimeSuffixAfterStable(displayText, stableText);
      } else {
        nextStable = "";
        nextRealtime = displayText;
      }
      if (shouldKeepCurrentLive(nextStable, nextRealtime)) {
        return;
      }
      stableText = nextStable;
      realtimeText = nextRealtime;
      renderTranscript();
    }

    async function postJson(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload || {})
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function sendVideoTime(path) {
      if (!activeSessionId) {
        return;
      }
      postJson(path, {
        session_id: activeSessionId,
        current_time: video.currentTime || 0
      }).catch((error) => appendStatus(error.message, "error"));
    }

    function startTimePings() {
      if (timePing) {
        clearInterval(timePing);
      }
      timePing = setInterval(() => {
        if (!video.paused && !video.ended) {
          sendVideoTime("/api/video-time");
        }
      }, 250);
    }

    function stopTimePings() {
      if (timePing) {
        clearInterval(timePing);
        timePing = null;
      }
    }

    video.addEventListener("playing", () => {
      placeholder.classList.add("hidden");
      if (!postedFirstPlaying) {
        postedFirstPlaying = true;
        appendStatus("Local video reports PLAYING. Audio feed may start.");
        sendVideoTime("/api/video-playing");
      }
      startTimePings();
    });

    video.addEventListener("pause", stopTimePings);
    video.addEventListener("ended", stopTimePings);

    const events = new EventSource("/events");
    events.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "");
    });
    events.addEventListener("error-status", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(data.message || "Error", "error");
    });
    events.addEventListener("media-ready", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus(`Local media ready: ${data.title || data.filename || "video"}`);
      video.src = data.media_url;
      video.load();
      video.play().catch((error) => {
        appendStatus(`Click the video play button to start: ${error.message}`, "error");
      });
    });
    events.addEventListener("realtime", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      splitRealtime(
        data.display_text || data.text || "",
        data.stable_text || "",
        data.unstable_text || ""
      );
    });
    events.addEventListener("final", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      const text = (data.text || "").trim();
      if (text) {
        const liveRemainder = livePartsAfterFinal(text);
        finalTexts.push(text);
        stableText = liveRemainder.stable;
        realtimeText = liveRemainder.realtime;
        renderTranscript();
      }
    });

    startButton.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (!url) {
        appendStatus("Paste a YouTube URL first.", "error");
        return;
      }

      startButton.disabled = true;
      resetDisplay();
      appendStatus("Started.");

      try {
        const data = await postJson("/api/start", {url});
        activeSessionId = data.session_id;
      } catch (error) {
        appendStatus(error.message, "error");
      } finally {
        startButton.disabled = false;
      }
    });

    stopButton.addEventListener("click", async () => {
      stopTimePings();
      video.pause();
      try {
        await postJson("/api/stop", {session_id: activeSessionId});
        appendStatus("Stop requested.");
      } catch (error) {
        appendStatus(error.message, "error");
      }
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Private YouTube + RealtimeSTT synchronized browser GUI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".youtube_sync_cache",
    )
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--rt-model", default="tiny.en")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--split-marks", default="sentence")
    parser.add_argument("--realtime-processing-pause", type=float, default=0.1)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--beam-size-realtime", type=int, default=1)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--play-timeout", type=float, default=90.0)
    parser.add_argument("--drain-seconds", type=float, default=5.0)
    parser.add_argument(
        "--yt-format",
        default=(
            "best[height<=720][ext=mp4][vcodec!=none][acodec!=none]/"
            "best[ext=mp4][vcodec!=none][acodec!=none]/"
            "best[vcodec!=none][acodec!=none]"
        ),
        help=(
            "Primary yt-dlp format selector. Defaults to progressive video "
            "with audio to avoid YouTube split-stream 403 failures."
        ),
    )
    parser.add_argument(
        "--youtube-client",
        default="web_safari,web",
        help=(
            "Comma-separated yt-dlp YouTube player clients. Use an empty value "
            "to let yt-dlp choose."
        ),
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Optional Netscape cookies.txt file for yt-dlp.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Optional browser cookie source for yt-dlp, for example chrome.",
    )
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        help="HTTP User-Agent forwarded to yt-dlp.",
    )
    return parser.parse_args()


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[queue.Queue[tuple[str, str]]] = set()
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue[tuple[str, str]]:
        subscriber: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[tuple[str, str]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        message = json_dumps(payload)
        with self._lock:
            subscribers = tuple(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber.put_nowait((event, message))
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait((event, message))
                except queue.Full:
                    pass


class YouTubeRealtimeController:
    def __init__(self, args: argparse.Namespace, bus: EventBus) -> None:
        self.args = args
        self.bus = bus
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._video_playing_event = threading.Event()
        self._latest_video_time = 0.0
        self._latest_video_time_at = 0.0
        self._ffmpeg_process: subprocess.Popen[bytes] | None = None
        self._recorder: Any = None
        self._media_paths: dict[str, Path] = {}

    def start(self, url: str) -> str:
        self.stop(emit=False)

        session_id = uuid.uuid4().hex
        stop_event = threading.Event()
        video_playing_event = threading.Event()

        with self._lock:
            self._session_id = session_id
            self._stop_event = stop_event
            self._video_playing_event = video_playing_event
            self._latest_video_time = 0.0
            self._latest_video_time_at = 0.0
            self._ffmpeg_process = None
            self._recorder = None
            self._worker = threading.Thread(
                target=self._run_session,
                args=(session_id, url, stop_event, video_playing_event),
                name="YouTubeRealtimeSTTWorker",
                daemon=True,
            )
            self._worker.start()

        self._status(session_id, "Started.")
        return session_id

    def stop(self, emit: bool = True) -> None:
        with self._lock:
            session_id = self._session_id
            stop_event = self._stop_event
            process = self._ffmpeg_process
            recorder = self._recorder

        stop_event.set()
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

        if recorder is not None:
            try:
                recorder.stop()
            except Exception:
                pass
            try:
                recorder.shutdown()
            except Exception:
                pass

        if emit and session_id:
            self._status(session_id, "Stop requested.")

    def update_video_time(self, session_id: str | None, current_time: Any) -> None:
        with self._lock:
            if session_id != self._session_id:
                return
            self._latest_video_time = max(0.0, float(current_time or 0.0))
            self._latest_video_time_at = time.monotonic()

    def mark_video_playing(self, session_id: str | None, current_time: Any) -> None:
        self.update_video_time(session_id, current_time)
        with self._lock:
            if session_id != self._session_id:
                return
            self._video_playing_event.set()
        self._status(
            session_id,
            "Local video is playing. Audio feed is allowed to start.",
        )

    def media_path(self, session_id: str) -> Path | None:
        with self._lock:
            return self._media_paths.get(session_id)

    def _is_current(self, session_id: str) -> bool:
        with self._lock:
            return session_id == self._session_id

    def _video_offset(self, session_id: str) -> float:
        with self._lock:
            if session_id != self._session_id:
                return 0.0
            if not self._latest_video_time_at:
                return self._latest_video_time
            elapsed = time.monotonic() - self._latest_video_time_at
            return max(0.0, self._latest_video_time + elapsed)

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit(
            "error-status",
            {"session_id": session_id, "message": message},
        )

    def _run_session(
            self,
            session_id: str,
            url: str,
            stop_event: threading.Event,
            video_playing_event: threading.Event,
    ) -> None:
        final_thread: threading.Thread | None = None
        recorder = None

        try:
            media_path, title = self._download_media(session_id, url, stop_event)
            if stop_event.is_set() or not self._is_current(session_id):
                return

            with self._lock:
                self._media_paths[session_id] = media_path

            recorder = self._create_recorder(session_id)
            with self._lock:
                if session_id != self._session_id:
                    return
                self._recorder = recorder

            recorder.start()
            final_thread = threading.Thread(
                target=self._consume_final_text,
                args=(session_id, recorder, stop_event),
                name="YouTubeRealtimeSTTFinalConsumer",
                daemon=True,
            )
            final_thread.start()

            self.bus.emit(
                "media-ready",
                {
                    "session_id": session_id,
                    "media_url": f"/media/{session_id}/{media_path.name}",
                    "filename": media_path.name,
                    "title": title,
                },
            )
            self._status(
                session_id,
                "Waiting for confirmed local video PLAYING state.",
            )
            if not video_playing_event.wait(timeout=self.args.play_timeout):
                raise TimeoutError(
                    "Timed out waiting for the browser video to report PLAYING."
                )

            offset = self._video_offset(session_id)
            self._status(
                session_id,
                f"Converting audio from local video timestamp {offset:.2f}s.",
            )
            self._feed_ffmpeg_audio(
                session_id,
                media_path,
                offset,
                recorder,
                stop_event,
            )

            if stop_event.is_set():
                return

            self._status(session_id, "Audio stream ended. Draining final text.")
            time.sleep(max(0.0, self.args.drain_seconds))
            try:
                recorder.stop()
            except Exception:
                pass
            time.sleep(0.5)

        except Exception as exc:
            if self._is_current(session_id):
                self._error(session_id, str(exc))
        finally:
            stop_event.set()
            if final_thread is not None:
                final_thread.join(timeout=2.0)
            if recorder is not None:
                try:
                    recorder.shutdown()
                except Exception:
                    pass
            if self._is_current(session_id):
                self._status(session_id, "Finished.")

    def _download_media(
            self,
            session_id: str,
            url: str,
            stop_event: threading.Event,
    ) -> tuple[Path, str]:
        self._status(session_id, "Retrieving YouTube video.")
        try:
            import yt_dlp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "yt-dlp is required. Install it in this environment first."
            ) from exc

        cache_dir = self.args.cache_dir.resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        last_progress_at = 0.0
        info = self._extract_video_info(yt_dlp, url)
        title = info.get("title") or info.get("id") or "YouTube video"
        cached_media = self._find_cached_media(cache_dir, info)
        if cached_media is not None:
            self._status(
                session_id,
                f"Reusing already downloaded media: {cached_media.name}",
            )
            return cached_media, title

        def progress_hook(data: dict[str, Any]) -> None:
            nonlocal last_progress_at
            if stop_event.is_set():
                raise RuntimeError("Download stopped.")
            status = data.get("status")
            now = time.monotonic()
            if status == "downloading" and now - last_progress_at >= 1.0:
                last_progress_at = now
                percent = (data.get("_percent_str") or "").strip()
                speed = (data.get("_speed_str") or "").strip()
                eta = (data.get("_eta_str") or "").strip()
                parts = [part for part in (percent, speed, f"eta {eta}" if eta else "") if part]
                self._status(session_id, "Downloading video: " + ", ".join(parts))
            elif status == "finished":
                self._status(session_id, "Download finished. Merging media.")

        download_attempts = [
            (
                "progressive MP4",
                self.args.yt_format,
            ),
            (
                "720p split MP4/M4A",
                "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
                "bv*[height<=720]+ba/best",
            ),
            (
                "yt-dlp default best",
                "best",
            ),
        ]

        last_error: Exception | None = None
        downloaded_info = None
        for label, format_selector in download_attempts:
            if stop_event.is_set():
                raise RuntimeError("Download stopped.")
            self._status(session_id, f"Download attempt: {label}.")
            self._remove_partial_downloads(cache_dir)
            try:
                info = self._download_with_yt_dlp(
                    yt_dlp,
                    url,
                    cache_dir,
                    format_selector,
                    progress_hook,
                )
                downloaded_info = info
                break
            except Exception as exc:
                last_error = exc
                message = str(exc).replace("\n", " ").strip()
                self._status(session_id, f"{label} failed: {message}")

        if downloaded_info is None:
            raise RuntimeError(
                "All yt-dlp download attempts failed. "
                "If this video requires an authenticated browser session, retry "
                "with --cookies-from-browser chrome. "
                f"Last error: {last_error}"
            )

        if "entries" in downloaded_info:
            entries = [entry for entry in downloaded_info.get("entries") or [] if entry]
            if not entries:
                raise RuntimeError("yt-dlp returned an empty playlist.")
            downloaded_info = entries[0]

        media_path = self._resolved_download_path(cache_dir, downloaded_info)
        if not media_path.exists():
            raise RuntimeError(f"Downloaded media file was not found: {media_path}")

        title = downloaded_info.get("title") or title or media_path.stem
        self._status(session_id, f"Local video ready: {media_path.name}")
        return media_path, title

    def _extract_video_info(self, yt_dlp: Any, url: str) -> dict[str, Any]:
        ydl_options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "extract_flat": False,
            "http_headers": {
                "User-Agent": self.args.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            },
        }
        clients = [
            client.strip()
            for client in (self.args.youtube_client or "").split(",")
            if client.strip()
        ]
        if clients:
            ydl_options["extractor_args"] = {
                "youtube": {"player_client": clients},
            }
        if self.args.cookies:
            ydl_options["cookiefile"] = str(self.args.cookies)
        if self.args.cookies_from_browser:
            ydl_options["cookiesfrombrowser"] = (
                self.args.cookies_from_browser,
                None,
                None,
                None,
            )

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=False)

        if "entries" in info:
            entries = [entry for entry in info.get("entries") or [] if entry]
            if not entries:
                raise RuntimeError("yt-dlp returned an empty playlist.")
            info = entries[0]
        return info

    def _find_cached_media(
            self,
            cache_dir: Path,
            info: dict[str, Any],
    ) -> Path | None:
        video_id = info.get("id")
        if not video_id:
            return None

        media_extensions = {
            ".mp4",
            ".m4v",
            ".webm",
            ".mkv",
            ".mov",
            ".avi",
            ".mp3",
            ".m4a",
            ".opus",
            ".ogg",
            ".wav",
        }
        matches = [
            path
            for path in cache_dir.glob(f"{video_id}.*")
            if (
                path.is_file()
                and path.suffix.lower() in media_extensions
                and not path.name.endswith(".part")
                and path.stat().st_size > 0
            )
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[0].resolve()

    def _download_with_yt_dlp(
            self,
            yt_dlp: Any,
            url: str,
            cache_dir: Path,
            format_selector: str,
            progress_hook: Any,
    ) -> dict[str, Any]:
        ydl_options: dict[str, Any] = {
            "format": format_selector,
            "merge_output_format": "mp4",
            "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
            "ffmpeg_location": self._ffmpeg_path(),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "progress_hooks": [progress_hook],
            "windowsfilenames": True,
            "continuedl": False,
            "overwrites": True,
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 5,
            "extractor_retries": 3,
            "http_headers": {
                "User-Agent": self.args.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
            },
        }

        clients = [
            client.strip()
            for client in (self.args.youtube_client or "").split(",")
            if client.strip()
        ]
        if clients:
            ydl_options["extractor_args"] = {
                "youtube": {"player_client": clients},
            }

        if self.args.cookies:
            ydl_options["cookiefile"] = str(self.args.cookies)

        if self.args.cookies_from_browser:
            ydl_options["cookiesfrombrowser"] = (
                self.args.cookies_from_browser,
                None,
                None,
                None,
            )

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            return ydl.extract_info(url, download=True)

    def _remove_partial_downloads(self, cache_dir: Path) -> None:
        for path in cache_dir.glob("*.part"):
            try:
                path.unlink()
            except OSError:
                pass

    def _resolved_download_path(self, cache_dir: Path, info: dict[str, Any]) -> Path:
        requested = info.get("requested_downloads") or []
        for item in requested:
            filepath = item.get("filepath")
            if filepath and Path(filepath).exists():
                return Path(filepath).resolve()

        video_id = info.get("id")
        if video_id:
            matches = sorted(
                cache_dir.glob(f"{video_id}.*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return matches[0].resolve()

        filename = info.get("_filename") or info.get("filename")
        if filename and Path(filename).exists():
            return Path(filename).resolve()

        raise RuntimeError("Could not determine the downloaded media filename.")

    def _create_recorder(self, session_id: str) -> Any:
        self._status(session_id, "Initializing RealtimeSTT.")

        if os.name == "nt":
            try:
                from torchaudio._extension.utils import _init_dll_path

                _init_dll_path()
            except Exception:
                pass

        from RealtimeSTT import AudioToTextRecorder

        def realtime_update(text: str) -> None:
            self.bus.emit(
                "realtime",
                {
                    "session_id": session_id,
                    "display_text": text or "",
                    "stable_text": "",
                },
            )

        def stabilization_update(event: Any) -> None:
            display_text = getattr(event, "display_text", None)
            if display_text is None:
                display_text = getattr(event, "raw_observation_text", "")
            stable_text = getattr(event, "stable_text", "") or ""
            unstable_text = getattr(event, "unstable_text", "") or ""
            self.bus.emit(
                "realtime",
                {
                    "session_id": session_id,
                    "display_text": display_text or "",
                    "stable_text": stable_text,
                    "unstable_text": unstable_text,
                },
            )

        recorder = AudioToTextRecorder(
            use_microphone=False,
            spinner=False,
            model=self.args.model,
            realtime_model_type=self.args.rt_model,
            language=self.args.language,
            device=self.args.device,
            compute_type=self.args.compute_type,
            download_root=self.args.download_root,
            enable_realtime_transcription=True,
            realtime_punctuation_split_marks=self.args.split_marks,
            realtime_processing_pause=self.args.realtime_processing_pause,
            init_realtime_after_seconds=0.0,
            on_realtime_transcription_update=None,
            on_realtime_text_stabilization_update=stabilization_update,
            min_length_of_recording=0,
            min_gap_between_recordings=0,
            post_speech_silence_duration=0.6,
            silero_sensitivity=0.05,
            webrtc_sensitivity=3,
            silero_deactivity_detection=True,
            realtime_transcription_use_syllable_boundaries=True,
            realtime_boundary_detector_sensitivity=0.6,
            realtime_boundary_followup_delays=(0.05, 0.2),
            beam_size=self.args.beam_size,
            beam_size_realtime=self.args.beam_size_realtime,
            no_log_file=True,
            faster_whisper_vad_filter=False,
        )

        self._status(session_id, "RealtimeSTT ready.")
        return recorder

    def _ffmpeg_path(self) -> str:
        configured = self.args.ffmpeg
        if Path(configured).exists():
            return configured
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        raise RuntimeError(
            f"FFmpeg executable not found: {configured}. "
            "Install FFmpeg or pass --ffmpeg."
        )

    def _feed_ffmpeg_audio(
            self,
            session_id: str,
            media_path: Path,
            offset: float,
            recorder: Any,
            stop_event: threading.Event,
    ) -> None:
        chunk_bytes = max(320, int(16000 * 2 * self.args.chunk_ms / 1000))
        command = [
            self._ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
        ]

        if offset > 0:
            command.extend(["-ss", f"{offset:.3f}"])

        command.extend([
            "-re",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ])

        self._status(session_id, "Feeding synchronized audio to RealtimeSTT.")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with self._lock:
            if session_id == self._session_id:
                self._ffmpeg_process = process

        assert process.stdout is not None
        while not stop_event.is_set():
            chunk = process.stdout.read(chunk_bytes)
            if not chunk:
                break
            recorder.feed_audio(chunk, original_sample_rate=16000)

        if stop_event.is_set() and process.poll() is None:
            process.terminate()

        return_code = process.wait(timeout=10)
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()[-2000:]

        with self._lock:
            if session_id == self._session_id:
                self._ffmpeg_process = None

        if return_code != 0 and not stop_event.is_set():
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed with code {return_code}: {detail}")

    def _consume_final_text(
            self,
            session_id: str,
            recorder: Any,
            stop_event: threading.Event,
    ) -> None:
        while True:
            queue_empty = recorder.recorded_audio_queue.empty()
            if stop_event.is_set() and queue_empty:
                return
            if queue_empty:
                time.sleep(0.05)
                continue

            try:
                text = (recorder.text() or "").strip()
            except Exception as exc:
                if not stop_event.is_set():
                    self._error(session_id, f"Final transcription failed: {exc}")
                return

            if text:
                self.bus.emit("final", {"session_id": session_id, "text": text})


class RequestHandler(BaseHTTPRequestHandler):
    server: "GuiServer"

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/events":
            self._serve_events()
            return
        if self.path.startswith("/media/"):
            self._serve_media()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                url = str(payload.get("url") or "").strip()
                if not url:
                    raise ValueError("Missing YouTube URL.")
                session_id = self.server.controller.start(url)
                self._send_json({"session_id": session_id})
                return

            if self.path == "/api/video-playing":
                self.server.controller.mark_video_playing(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return

            if self.path == "/api/video-time":
                self.server.controller.update_video_time(
                    payload.get("session_id"),
                    payload.get("current_time"),
                )
                self._send_json({"ok": True})
                return

            if self.path == "/api/stop":
                self.server.controller.stop()
                self._send_json({"ok": True})
                return

            self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json_dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: int = 200,
            extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        subscriber = self.server.bus.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                try:
                    event, payload = subscriber.get(timeout=10)
                    message = f"event: {event}\ndata: {payload}\n\n"
                except queue.Empty:
                    message = ": heartbeat\n\n"
                self.wfile.write(message.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass
        finally:
            self.server.bus.unsubscribe(subscriber)

    def _serve_media(self) -> None:
        match = re.match(r"^/media/([a-f0-9]+)/", self.path)
        if not match:
            self.send_error(404)
            return

        media_path = self.server.controller.media_path(match.group(1))
        if media_path is None or not media_path.exists():
            self.send_error(404)
            return

        file_size = media_path.stat().st_size
        content_type = mimetypes.guess_type(media_path.name)[0] or "video/mp4"
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self._copy_file_range(media_path, 0, file_size - 1)
            return

        range_match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not range_match:
            self.send_error(416)
            return

        start_text, end_text = range_match.groups()
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        length = end - start + 1

        self.send_response(206)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        self._copy_file_range(media_path, start, end)

    def _copy_file_range(self, media_path: Path, start: int, end: int) -> None:
        remaining = end - start + 1
        with media_path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class GuiServer(ThreadingHTTPServer):
    def __init__(
            self,
            address: tuple[str, int],
            args: argparse.Namespace,
            bus: EventBus,
    ) -> None:
        super().__init__(address, RequestHandler)
        self.args = args
        self.bus = bus
        self.controller = YouTubeRealtimeController(args, bus)


def main() -> int:
    args = parse_args()
    bus = EventBus()
    server = GuiServer((args.host, args.port), args, bus)
    url = f"http://{args.host}:{server.server_port}/"

    print(f"Serving RealtimeSTT YouTube sync GUI at {url}")
    print(f"Media cache: {args.cache_dir.resolve()}")
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.controller.stop(emit=False)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
