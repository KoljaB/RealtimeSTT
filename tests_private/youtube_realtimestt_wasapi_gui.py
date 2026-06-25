"""Private YouTube iframe + WASAPI loopback RealtimeSTT GUI.

This variant does not download or feed YouTube media. The browser plays the
YouTube iframe directly, and RealtimeSTT captures the resulting system audio
through a WASAPI loopback-style input device.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import queue
import re
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RealtimeSTT YouTube WASAPI</title>
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
      min-height: 280px;
      position: relative;
    }

    #player, .placeholder {
      width: 100%;
      height: 100%;
      min-height: 280px;
    }

    .placeholder {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 18px;
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
        <div id="player"><div class="placeholder">YouTube video</div></div>
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

  <script src="https://www.youtube.com/iframe_api"></script>
  <script>
    const urlInput = document.getElementById("urlInput");
    const startButton = document.getElementById("startButton");
    const stopButton = document.getElementById("stopButton");
    const statusLog = document.getElementById("statusLog");
    const transcript = document.getElementById("transcript");

    let player = null;
    let ytReady = false;
    let pendingVideoId = null;
    let activeSessionId = null;
    let timePing = null;
    let finalTexts = [];
    let stableText = "";
    let realtimeText = "";
    let lastVisibleTranscriptHtml = "";
    let sawStructuredRealtime = false;
    let postedFirstPlaying = false;

    function snippet(text, limit = 260) {
      text = text || "";
      if (text.length <= limit) {
        return text;
      }
      return `${text.slice(0, limit)}...`;
    }

    function debugLog(stage, payload) {
      const body = JSON.stringify({
        stage,
        session_id: activeSessionId,
        browser_time_ms: performance.now(),
        payload: payload || {}
      });
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([body], {type: "application/json"});
          navigator.sendBeacon("/api/debug-log", blob);
          return;
        }
      } catch (_) {
      }
      fetch("/api/debug-log", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body,
        keepalive: true
      }).catch(() => {});
    }

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
      renderTranscript();
    }

    function renderTranscript() {
      const beforeText = transcript.textContent || "";
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

      let usedCache = false;
      if (transcript.textContent.trim()) {
        lastVisibleTranscriptHtml = transcript.innerHTML;
      } else if (lastVisibleTranscriptHtml) {
        transcript.innerHTML = lastVisibleTranscriptHtml;
        usedCache = true;
      }

      transcript.scrollTop = transcript.scrollHeight;
      debugLog("render", {
        before_len: beforeText.trim().length,
        rendered_len: (transcript.textContent || "").trim().length,
        used_cache: usedCache,
        final_count: finalTexts.length,
        stable_len: stableText.length,
        realtime_len: realtimeText.length,
        stable: snippet(stableText),
        realtime: snippet(realtimeText),
        rendered: snippet(transcript.textContent || "")
      });
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
      if (liveNorm.includes(finalNorm)) {
        return live;
      }
      return live;
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
      const beforeStable = stableText;
      const beforeRealtime = realtimeText;
      displayText = displayText || "";
      stable = stable || "";
      unstable = unstable || "";
      const hasStructuredParts = Boolean(stable || unstable);

      if (!displayText && !stable && !unstable) {
        debugLog("splitRealtime", {
          decision: "ignored_empty",
          before_stable_len: beforeStable.length,
          before_realtime_len: beforeRealtime.length
        });
        return;
      }

      if (!hasStructuredParts && sawStructuredRealtime) {
        debugLog("splitRealtime", {
          decision: "ignored_legacy_after_structured",
          input_display_len: displayText.length,
          before_stable_len: beforeStable.length,
          before_realtime_len: beforeRealtime.length,
          input_display: snippet(displayText),
          before_stable: snippet(beforeStable),
          before_realtime: snippet(beforeRealtime)
        });
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
        debugLog("splitRealtime", {
          decision: "kept_current_live",
          input_display_len: displayText.length,
          input_stable_len: stable.length,
          input_unstable_len: unstable.length,
          before_stable_len: beforeStable.length,
          before_realtime_len: beforeRealtime.length,
          next_stable_len: nextStable.length,
          next_realtime_len: nextRealtime.length,
          input_display: snippet(displayText),
          input_stable: snippet(stable),
          input_unstable: snippet(unstable),
          before_stable: snippet(beforeStable),
          before_realtime: snippet(beforeRealtime),
          next_stable: snippet(nextStable),
          next_realtime: snippet(nextRealtime)
        });
        return;
      }
      stableText = nextStable;
      realtimeText = nextRealtime;
      debugLog("splitRealtime", {
        decision: "applied",
        input_display_len: displayText.length,
        input_stable_len: stable.length,
        input_unstable_len: unstable.length,
        before_stable_len: beforeStable.length,
        before_realtime_len: beforeRealtime.length,
        next_stable_len: nextStable.length,
        next_realtime_len: nextRealtime.length,
        input_display: snippet(displayText),
        input_stable: snippet(stable),
        input_unstable: snippet(unstable),
        before_stable: snippet(beforeStable),
        before_realtime: snippet(beforeRealtime),
        next_stable: snippet(nextStable),
        next_realtime: snippet(nextRealtime)
      });
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

    function playerCurrentTime() {
      try {
        return player ? player.getCurrentTime() || 0 : 0;
      } catch (_) {
        return 0;
      }
    }

    function sendVideoTime(path) {
      if (!activeSessionId || !player) {
        return;
      }
      postJson(path, {
        session_id: activeSessionId,
        current_time: playerCurrentTime()
      }).catch((error) => appendStatus(error.message, "error"));
    }

    function startTimePings() {
      if (timePing) {
        clearInterval(timePing);
      }
      timePing = setInterval(() => {
        if (player && player.getPlayerState && player.getPlayerState() === YT.PlayerState.PLAYING) {
          sendVideoTime("/api/video-time");
        }
      }, 500);
    }

    function stopTimePings() {
      if (timePing) {
        clearInterval(timePing);
        timePing = null;
      }
    }

    function createPlayer(videoId) {
      pendingVideoId = null;
      if (player && player.destroy) {
        player.destroy();
      }
      player = new YT.Player("player", {
        width: "100%",
        height: "100%",
        videoId,
        playerVars: {
          autoplay: 1,
          controls: 1,
          rel: 0,
          modestbranding: 1,
          playsinline: 1
        },
        events: {
          onReady: (event) => {
            appendStatus("YouTube player ready. Starting playback.");
            event.target.playVideo();
          },
          onStateChange: (event) => {
            if (event.data === YT.PlayerState.PLAYING) {
              if (!postedFirstPlaying) {
                postedFirstPlaying = true;
                appendStatus("YouTube reports PLAYING. WASAPI capture is already armed.");
                sendVideoTime("/api/video-playing");
              }
              startTimePings();
            } else if (
              event.data === YT.PlayerState.PAUSED ||
              event.data === YT.PlayerState.ENDED
            ) {
              stopTimePings();
            }
          }
        }
      });
    }

    window.onYouTubeIframeAPIReady = function() {
      ytReady = true;
      if (pendingVideoId) {
        createPlayer(pendingVideoId);
      }
    };

    function loadVideo(videoId) {
      if (!ytReady) {
        pendingVideoId = videoId;
        appendStatus("Waiting for YouTube player API.");
        return;
      }
      createPlayer(videoId);
    }

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
    events.addEventListener("capture-ready", (event) => {
      const data = JSON.parse(event.data);
      if (data.session_id && activeSessionId && data.session_id !== activeSessionId) {
        return;
      }
      appendStatus("Capture ready. Loading YouTube video.");
      loadVideo(data.video_id);
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
        debugLog("finalEvent", {
          final_len: text.length,
          live_before_stable_len: stableText.length,
          live_before_realtime_len: realtimeText.length,
          live_remainder_len: `${liveRemainder.stable}${liveRemainder.realtime}`.length,
          stable_remainder_len: liveRemainder.stable.length,
          realtime_remainder_len: liveRemainder.realtime.length,
          final: snippet(text),
          live_before_stable: snippet(stableText),
          live_before_realtime: snippet(realtimeText),
          live_remainder: snippet(`${liveRemainder.stable}${liveRemainder.realtime}`),
          stable_remainder: snippet(liveRemainder.stable),
          realtime_remainder: snippet(liveRemainder.realtime)
        });
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
      if (player && player.pauseVideo) {
        player.pauseVideo();
      }
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
        description="Private YouTube iframe + WASAPI loopback RealtimeSTT GUI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--input-device-index", type=int, default=None)
    parser.add_argument("--allow-default-input", action="store_true")
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
    parser.add_argument(
        "--trace-log",
        type=Path,
        default=None,
        help="Optional JSONL trace path for backend/frontend UI events.",
    )
    parser.add_argument(
        "--analyze-trace",
        type=Path,
        default=None,
        help="Analyze a previously captured JSONL trace and exit.",
    )
    return parser.parse_args()


def extract_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    query_video_id = parse_qs(parsed.query).get("v", [""])[0]
    if query_video_id:
        return query_video_id

    path_match = re.search(r"/(?:embed|shorts|live)/([^/?#]+)", parsed.path)
    if path_match:
        return path_match.group(1)

    raise ValueError("Could not extract a YouTube video id from the URL.")


def json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class TraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
            self,
            source: str,
            event: str,
            payload: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "time": time.time(),
            "source": source,
            "event": event,
            "payload": payload or {},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def list_audio_input_devices() -> list[dict[str, Any]]:
    try:
        import pyaudiowpatch as pyaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyaudiowpatch is required for WASAPI loopback capture."
        ) from exc

    audio = pyaudio.PyAudio()
    devices: list[dict[str, Any]] = []
    try:
        host_api_names = {}
        for host_index in range(audio.get_host_api_count()):
            try:
                host_info = audio.get_host_api_info_by_index(host_index)
                host_api_names[host_index] = host_info.get("name", "")
            except Exception:
                host_api_names[host_index] = ""

        for index in range(audio.get_device_count()):
            try:
                info = audio.get_device_info_by_index(index)
            except Exception:
                continue
            max_input_channels = int(info.get("maxInputChannels") or 0)
            if max_input_channels <= 0:
                continue
            host_api_index = int(info.get("hostApi") or 0)
            devices.append({
                "index": int(info.get("index", index)),
                "name": str(info.get("name", "")),
                "host_api": host_api_names.get(host_api_index, ""),
                "channels": max_input_channels,
                "sample_rate": int(float(info.get("defaultSampleRate") or 0)),
                "is_loopback": bool(info.get("isLoopbackDevice")),
            })
    finally:
        audio.terminate()

    return devices


def choose_wasapi_loopback_device(
        requested_index: int | None,
        allow_default_input: bool,
) -> tuple[int | None, dict[str, Any] | None, list[dict[str, Any]]]:
    devices = list_audio_input_devices()
    if requested_index is not None:
        for device in devices:
            if device["index"] == requested_index:
                return requested_index, device, devices
        raise RuntimeError(f"Input device index {requested_index} is not available.")

    try:
        import pyaudiowpatch as pyaudio

        audio = pyaudio.PyAudio()
        try:
            default_loopback = audio.get_default_wasapi_loopback()
        finally:
            audio.terminate()
        if default_loopback and default_loopback.get("isLoopbackDevice"):
            index = int(default_loopback["index"])
            for device in devices:
                if device["index"] == index:
                    return index, device, devices
            return index, {
                "index": index,
                "name": str(default_loopback.get("name", "")),
                "host_api": "Windows WASAPI",
                "channels": int(default_loopback.get("maxInputChannels") or 2),
                "sample_rate": int(float(default_loopback.get("defaultSampleRate") or 48000)),
                "is_loopback": True,
            }, devices
    except Exception:
        pass

    loopbacks = [device for device in devices if device.get("is_loopback")]
    if loopbacks:
        return loopbacks[0]["index"], loopbacks[0], devices

    if allow_default_input:
        return None, None, devices

    device_lines = [
        f"{item['index']}: {item['name']} [{item['host_api']}, "
        f"{item['channels']}ch, {item['sample_rate']} Hz"
        f"{', loopback' if item.get('is_loopback') else ''}]"
        for item in devices
    ]
    raise RuntimeError(
        "No WASAPI loopback-like input device was detected. "
        "Rerun with --input-device-index N after choosing one of:\n"
        + "\n".join(device_lines)
    )


class EventBus:
    def __init__(self, trace: TraceLogger | None = None) -> None:
        self._subscribers: set[queue.Queue[tuple[str, str]]] = set()
        self._lock = threading.Lock()
        self._trace = trace

    def subscribe(self) -> queue.Queue[tuple[str, str]]:
        subscriber: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=200)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[tuple[str, str]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._trace is not None:
            self._trace.write("backend_sse", event, payload)
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


class YouTubeWasapiController:
    def __init__(self, args: argparse.Namespace, bus: EventBus) -> None:
        self.args = args
        self.bus = bus
        self._lock = threading.RLock()
        self._session_id: str | None = None
        self._recorder: Any = None
        self._audio_interface: Any = None
        self._audio_stream: Any = None
        self._final_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._video_ids: dict[str, str] = {}

    def start(self, url: str) -> tuple[str, str]:
        video_id = extract_youtube_video_id(url)
        self.stop(emit=False)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._session_id = session_id
            self._stop_event = threading.Event()
            self._video_ids[session_id] = video_id
        self._status(session_id, "Started.")
        worker = threading.Thread(
            target=self._start_capture,
            args=(session_id,),
            name="YouTubeWasapiCaptureStarter",
            daemon=True,
        )
        worker.start()
        return session_id, video_id

    def stop(self, emit: bool = True) -> None:
        with self._lock:
            session_id = self._session_id
            recorder = self._recorder
            stream = self._audio_stream
            audio_interface = self._audio_interface
            stop_event = self._stop_event

        stop_event.set()
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio_interface is not None:
            try:
                audio_interface.terminate()
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

    def mark_video_playing(self, session_id: str | None, _current_time: Any) -> None:
        if not session_id:
            return
        with self._lock:
            if session_id != self._session_id:
                return
        self._status(session_id, "YouTube playback confirmed.")

    def update_video_time(self, _session_id: str | None, _current_time: Any) -> None:
        return

    def _is_current(self, session_id: str) -> bool:
        with self._lock:
            return session_id == self._session_id

    def _status(self, session_id: str | None, message: str) -> None:
        self.bus.emit("status", {"session_id": session_id, "message": message})

    def _error(self, session_id: str | None, message: str) -> None:
        self.bus.emit(
            "error-status",
            {"session_id": session_id, "message": message},
        )

    def _start_capture(self, session_id: str) -> None:
        try:
            self._status(session_id, "Selecting WASAPI loopback input.")
            input_device_index, device, devices = choose_wasapi_loopback_device(
                self.args.input_device_index,
                self.args.allow_default_input,
            )
            if device is not None:
                self._status(
                    session_id,
                    "Using input device "
                    f"{device['index']}: {device['name']} [{device['host_api']}].",
                )
            else:
                self._status(session_id, "Using the default input device.")
            if devices:
                self._status(
                    session_id,
                    "Available input devices: "
                    + "; ".join(
                        f"{item['index']}={item['name']} [{item['host_api']}]"
                        for item in devices
                    ),
                )

            recorder = self._create_recorder(session_id, input_device_index)
            audio_interface, stream = self._open_loopback_stream(
                session_id,
                input_device_index,
                recorder,
                device,
            )
            with self._lock:
                if session_id != self._session_id:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    try:
                        audio_interface.terminate()
                    except Exception:
                        pass
                    recorder.shutdown()
                    return
                self._recorder = recorder
                self._audio_interface = audio_interface
                self._audio_stream = stream

            recorder.start()
            self._status(session_id, "WASAPI capture started.")
            final_thread = threading.Thread(
                target=self._consume_final_text,
                args=(session_id, recorder, self._stop_event),
                name="YouTubeWasapiFinalConsumer",
                daemon=True,
            )
            with self._lock:
                self._final_thread = final_thread
            final_thread.start()
            video_id = self._video_ids.get(session_id)
            if video_id:
                self.bus.emit(
                    "capture-ready",
                    {"session_id": session_id, "video_id": video_id},
                )
        except Exception as exc:
            if self._is_current(session_id):
                self._error(session_id, str(exc))

    def _open_loopback_stream(
            self,
            session_id: str,
            input_device_index: int | None,
            recorder: Any,
            device: dict[str, Any] | None,
    ) -> tuple[Any, Any]:
        try:
            import numpy as np
            import pyaudiowpatch as pyaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyaudiowpatch and numpy are required for WASAPI loopback capture."
            ) from exc

        audio_interface = pyaudio.PyAudio()
        try:
            if input_device_index is None:
                device_info = audio_interface.get_default_wasapi_loopback()
                input_device_index = int(device_info["index"])
            else:
                device_info = audio_interface.get_device_info_by_index(
                    input_device_index
                )

            if not device_info.get("isLoopbackDevice"):
                raise RuntimeError(
                    f"Device {input_device_index} is not a WASAPI loopback device."
                )

            sample_rate = int(device_info["defaultSampleRate"])
            channels = int(device_info["maxInputChannels"])
            if channels <= 0:
                raise RuntimeError(
                    f"Loopback device {input_device_index} has no input channels."
                )

            display_name = (
                device["name"]
                if device is not None
                else str(device_info.get("name", input_device_index))
            )
            self._status(
                session_id,
                f"Opening loopback stream: {display_name}, "
                f"{channels}ch, {sample_rate} Hz.",
            )

            def feed(data: bytes, _frame_count: int, _time_info: Any, _status: Any):
                audio = np.frombuffer(data, np.int16)
                if channels > 1:
                    audio = audio.reshape(-1, channels)
                recorder.feed_audio(audio, original_sample_rate=sample_rate)
                return (None, pyaudio.paContinue)

            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=max(1, sample_rate // 10),
                stream_callback=feed,
            )
            stream.start_stream()
            return audio_interface, stream
        except Exception:
            try:
                audio_interface.terminate()
            except Exception:
                pass
            raise

    def _create_recorder(self, session_id: str, input_device_index: int | None) -> Any:
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
                    "unstable_text": "",
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
            input_device_index=None,
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

    def _consume_final_text(
            self,
            session_id: str,
            recorder: Any,
            stop_event: threading.Event,
    ) -> None:
        while not stop_event.is_set():
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
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/start":
                url = str(payload.get("url") or "").strip()
                if not url:
                    raise ValueError("Missing YouTube URL.")
                session_id, video_id = self.server.controller.start(url)
                self._send_json({"session_id": session_id, "video_id": video_id})
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
            if self.path == "/api/debug-log":
                self.server.trace.write(
                    "frontend",
                    str(payload.get("stage") or "debug"),
                    payload,
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
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self._send_bytes(
            json_dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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


class GuiServer(ThreadingHTTPServer):
    def __init__(
            self,
            address: tuple[str, int],
            args: argparse.Namespace,
            bus: EventBus,
            trace: TraceLogger,
    ) -> None:
        super().__init__(address, RequestHandler)
        self.args = args
        self.bus = bus
        self.trace = trace
        self.controller = YouTubeWasapiController(args, bus)


def analyze_trace(path: Path) -> int:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No trace records found in {path}")
        return 1

    start = records[0].get("time", 0)
    print(f"Trace: {path}")
    print(f"Records: {len(records)}")

    interesting = {
        "render",
        "splitRealtime",
        "finalEvent",
        "realtime",
        "final",
        "capture-ready",
        "error-status",
    }
    for record in records:
        event = record.get("event")
        payload = record.get("payload") or {}
        if event not in interesting:
            continue
        t = float(record.get("time", start) or start) - float(start or 0)
        if record.get("source") == "frontend":
            data = payload.get("payload") or {}
            if event == "render":
                print(
                    f"{t:8.3f}s frontend render "
                    f"rendered_len={data.get('rendered_len')} "
                    f"used_cache={data.get('used_cache')} "
                    f"final_count={data.get('final_count')} "
                    f"stable_len={data.get('stable_len')} "
                    f"rt_len={data.get('realtime_len')} "
                    f"rendered={data.get('rendered')!r}"
                )
            elif event == "splitRealtime":
                print(
                    f"{t:8.3f}s frontend split "
                    f"decision={data.get('decision')} "
                    f"in=({data.get('input_display_len')},"
                    f"{data.get('input_stable_len')},"
                    f"{data.get('input_unstable_len')}) "
                    f"next=({data.get('next_stable_len')},"
                    f"{data.get('next_realtime_len')}) "
                    f"display={data.get('input_display')!r}"
                )
            elif event == "finalEvent":
                print(
                    f"{t:8.3f}s frontend finalEvent "
                    f"final_len={data.get('final_len')} "
                    f"live_before=({data.get('live_before_stable_len')},"
                    f"{data.get('live_before_realtime_len')}) "
                    f"remainder={data.get('live_remainder_len')} "
                    f"final={data.get('final')!r}"
                )
        else:
            if event == "realtime":
                print(
                    f"{t:8.3f}s backend realtime "
                    f"display_len={len(payload.get('display_text') or '')} "
                    f"stable_len={len(payload.get('stable_text') or '')} "
                    f"unstable_len={len(payload.get('unstable_text') or '')} "
                    f"display={(payload.get('display_text') or '')[:140]!r}"
                )
            elif event == "final":
                print(
                    f"{t:8.3f}s backend final "
                    f"len={len(payload.get('text') or '')} "
                    f"text={(payload.get('text') or '')[:180]!r}"
                )
            else:
                print(f"{t:8.3f}s backend {event} {payload}")

    return 0


def main() -> int:
    args = parse_args()
    if args.analyze_trace is not None:
        return analyze_trace(args.analyze_trace)

    trace_path = args.trace_log
    if trace_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        trace_path = (
            Path(__file__).resolve().parent
            / ".youtube_wasapi_traces"
            / f"trace-{stamp}.jsonl"
        )
    trace = TraceLogger(trace_path)
    bus = EventBus(trace)
    server = GuiServer((args.host, args.port), args, bus, trace)
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Serving RealtimeSTT YouTube WASAPI GUI at {url}")
    print(f"Trace log: {trace.path}")
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
