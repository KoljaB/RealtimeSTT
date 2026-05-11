#!/usr/bin/env python3
import asyncio, websockets, pyaudio, numpy as np, json, struct, time, threading, sys, os, traceback
from datetime import datetime

# ========= EDIT YOUR WS URLS HERE =========
CONTROL_URL = ""      # optional, not used here
DATA_URL    = ""
# ==========================================

SAMPLE_RATE = 16000
CHUNK_SIZES = [4096, 2048, 1024]    # adaptive fallbacks when reconnecting
PING_INTERVAL = 20
PING_TIMEOUT  = 20

# --- Detect environment issues for global hotkeys (Wayland/headless)
def env_blocks_global_hotkeys():
    # 1) No display / SSH-only / headless
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
        # 2) Wayland blocks global key capture for most libs
        if os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False

def try_import_pynput():
    try:
        from pynput import keyboard as _kb
        return _kb
    except Exception:
        return None

class PTTClient:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.ws = None
        self.control_ws = None
        self.loop = None
        self.running = False
        self.stop_threads = False

        self.chunk_index = 0
        self.current_chunk = CHUNK_SIZES[self.chunk_index]

        self.ptt_active = False     # True while speaking (push or toggle)
        self.toggle_mode = False    # fallback if global hooks not available
        self.chunks_sent = 0
        self.start_time = None

        self.kb = try_import_pynput()
        self.block_global = env_blocks_global_hotkeys()
        if self.block_global or not self.kb:
            self.toggle_mode = True

    # ---------- Audio ----------
    def open_stream(self):
        if self.stream:
            return
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.current_chunk
        )

    def close_stream(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def cleanup_audio(self):
        self.close_stream()
        try:
            self.p.terminate()
        except Exception:
            pass

    # ---------- Framing ----------
    @staticmethod
    def checksum_int16(audio_bytes: bytes) -> int:
        arr = np.frombuffer(audio_bytes, dtype=np.int16)
        return int(np.sum(arr, dtype=np.int64)) & 0xFFFFFFFF

    def frame(self, audio_bytes: bytes) -> bytes:
        meta = {
            "sampleRate": SAMPLE_RATE,
            "dataLength": len(audio_bytes) // 2,
            "checksum": self.checksum_int16(audio_bytes),
            "timestamp": int(time.time() * 1000),
            "server_sent_to_stt": True
        }
        meta_json = json.dumps(meta).encode("utf-8")
        meta_len  = struct.pack("<I", len(meta_json))
        return meta_len + meta_json + audio_bytes

    # ---------- Networking ----------
    async def connect_ws(self):
        await self.disconnect_ws()
        self.ws = await websockets.connect(DATA_URL, ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)
        print("✅ Connected to data server")
        # Try to open control channel (non-fatal if it fails)
        await self.connect_control()

    async def disconnect_ws(self):
        try:
            if self.ws:
                await self.ws.close()
        except Exception:
            pass
        self.ws = None

    async def connect_control(self):
        await self.disconnect_control()
        try:
            self.control_ws = await websockets.connect(CONTROL_URL, ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)
            print("✅ Connected to control server")
        except Exception as e:
            self.control_ws = None
            print(f"⚠️ Control connection unavailable: {e}")

    async def disconnect_control(self):
        try:
            if self.control_ws:
                await self.control_ws.close()
        except Exception:
            pass
        self.control_ws = None

    async def control_call(self, payload: dict):
        if not self.control_ws:
            return
        try:
            await self.control_ws.send(json.dumps(payload))
        except Exception:
            pass

    async def control_stop_and_clear(self):
        # Ask server to finalize current recording immediately
        await self.control_call({"command": "call_method", "method": "stop"})
        await self.control_call({"command": "call_method", "method": "clear_audio_queue"})

    async def receiver(self):
        try:
            async for msg in self.ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = datetime.now().strftime("%H:%M:%S")

                if data.get("type") == "error" and data.get("error") == "data_corruption":
                    print(f"\n[{t}] 🚨 Rejected due to data corruption: {data.get('message','')}")
                    self.running = False
                    break

                if data.get("type") == "fullSentence":
                    print(f"\n[{t}] ✅ Final: {data.get('text','')}")
                elif data.get("type") == "recording_start":
                    print(f"\n[{t}] 🔴 recording_start")
                elif data.get("type") == "recording_stop":
                    print(f"\n[{t}] ⏹️ recording_stop")
                elif data.get("type") not in [
                    "realtime","vad_detect_start","vad_detect_stop","transcription_start",
                    "start_turn_detection","stop_turn_detection","wakeword_detected",
                    "wakeword_detection_start","wakeword_detection_end"
                ]:
                    print(f"\n[{t}] 📨 {data}")
        except websockets.exceptions.ConnectionClosed:
            print("🔌 Data connection closed")
            self.running = False
        except Exception as e:
            print(f"❌ Receiver error: {e}")
            traceback.print_exc()
            self.running = False

    def audio_worker(self):
        print("⌨️  Push-to-talk:", "TOGGLE (Space to start/stop)" if self.toggle_mode else "HOLD SPACE to talk")
        was_active = False
        while not self.stop_threads and self.running:
            if not self.ptt_active:
                if was_active:
                    # Just turned OFF: send a short silence tail to help VAD close,
                    # then instruct server to stop & clear immediately via control WS.
                    try:
                        if self.ws:
                            silent = b"\x00" * (self.current_chunk * 2)  # int16 -> 2 bytes per sample
                            # send a couple of silent chunks
                            for _ in range(2):
                                framed = self.frame(silent)
                                asyncio.run_coroutine_threadsafe(self.ws.send(framed), self.loop)
                    except Exception:
                        pass
                    # Ask server to finalize this turn
                    try:
                        if self.loop:
                            asyncio.run_coroutine_threadsafe(self.control_stop_and_clear(), self.loop)
                    except Exception:
                        pass
                    self.close_stream()
                    was_active = False
                time.sleep(0.01)
                continue

            if not was_active:
                try:
                    self.open_stream()
                    was_active = True
                except Exception as e:
                    print(f"❌ Unable to open mic: {e}")
                    self.running = False
                    break

            try:
                audio_bytes = self.stream.read(self.current_chunk, exception_on_overflow=False)
                framed = self.frame(audio_bytes)
                if self.ws and self.running:
                    asyncio.run_coroutine_threadsafe(self.ws.send(framed), self.loop)
                    self.chunks_sent += 1
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Connection closed while sending")
                self.running = False
                break
            except Exception as e:
                print(f"⚠️ Audio send error: {e}")
                time.sleep(0.02)

    # ---------- Controls ----------
    def start_hotkeys(self):
        # Global push-to-talk (Space press/hold) when supported
        if self.toggle_mode:
            threading.Thread(target=self.toggle_stdin_loop, daemon=True).start()
            return

        # pynput global listener
        def on_press(key):
            try:
                if key == self.kb.Key.space:
                    if not self.ptt_active:
                        self.ptt_active = True
                        print("🎙️  PTT: ON")
                # Allow 'q' to quit when using global hotkeys
                try:
                    if hasattr(key, 'char') and key.char in ('q', 'Q'):
                        self.running = False
                except Exception:
                    pass
            except Exception:
                pass

        def on_release(key):
            try:
                if key == self.kb.Key.space:
                    if self.ptt_active:
                        self.ptt_active = False
                        print("🔇 PTT: OFF")
                        # Immediately tell server to stop & clear on key release
                        try:
                            if self.loop:
                                asyncio.run_coroutine_threadsafe(self.control_stop_and_clear(), self.loop)
                        except Exception:
                            pass
            except Exception:
                pass

        listener = self.kb.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()

    def toggle_stdin_loop(self):
        """
        Fallback: terminal-local toggle mode (works on SSH/headless/Wayland).
        Press SPACE to toggle ON/OFF. Press 'q' to quit.
        Runs in its own thread.
        """
        print("🧰 Fallback key mode (terminal): SPACE = toggle mic, q = quit")
        print("   Make sure this terminal has focus.")

        try:
            if os.name == "nt":
                # ---- Windows (unchanged) ----
                import msvcrt
                while not self.stop_threads:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch == ' ':
                            self.ptt_active = not self.ptt_active
                            print("🎙️  PTT: ON" if self.ptt_active else "🔇 PTT: OFF")
                        elif ch in ('q', 'Q'):
                            self.running = False
                            break
                    time.sleep(0.03)
                return

            # ---- Unix: make stdin non-echoing, non-canonical and poll with select ----
            import termios, tty, select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)

            # Start from current settings, then:
            new = termios.tcgetattr(fd)
            # lflags: turn off canonical mode (ICANON) and echo (ECHO)
            new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new)

            try:
                # Non-blocking read loop
                while not self.stop_threads:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if rlist:
                        ch = os.read(fd, 1).decode(errors='ignore')
                        if ch == ' ':
                            self.ptt_active = not self.ptt_active
                            print("🎙️  PTT: ON" if self.ptt_active else "🔇 PTT: OFF")
                        elif ch.lower() == 'q':
                            self.running = False
                            break
            finally:
                # Restore original terminal settings so your shell behaves normally
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

        except Exception as e:
            print(f"⚠️ Toggle input not available: {e}")
            print("   Use VAD or start/stop via control API instead.")

    async def one_session(self):
        self.loop = asyncio.get_event_loop()
        await self.connect_ws()
        self.running = True
        if self.start_time is None:
            self.start_time = time.time()

        recv_task = asyncio.create_task(self.receiver())
        audio_thr = threading.Thread(target=self.audio_worker, daemon=True)
        audio_thr.start()

        try:
            # No local key handling here; input is handled by global hotkeys (pynput)
            # or the terminal toggle thread (toggle_stdin_loop).
            while self.running:
                await asyncio.sleep(0.05)
        finally:
            try:
                recv_task.cancel()
            except Exception:
                pass
            await self.disconnect_control()
            await self.disconnect_ws()
            self.close_stream()
            audio_thr.join(timeout=1.0)
    async def run(self):
        print("=" * 50)
        print("🎯 RealtimeSTT Client (PTT with fallback)")
        print(f"   Data URL: {DATA_URL}")
        print(f"   SampleRate: {SAMPLE_RATE} Hz")
        print(f"   Chunks: {CHUNK_SIZES}")
        print("   Server policy: verify + reject on first corruption (threshold 0)")
        if self.toggle_mode:
            print("   Input mode: TOGGLE (terminal) — Space toggles ON/OFF")
        else:
            print("   Input mode: PUSH-TO-TALK (global Space press/hold)")
        print("=" * 50)

        self.start_hotkeys()

        retries = 0
        max_retries = 6
        base_backoff = 1.0

        while retries <= max_retries:
            try:
                print(f"🔧 Using chunk size: {self.current_chunk}")
                await self.one_session()
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt")
                break
            except Exception as e:
                print(f"❌ Session error: {e}")
                traceback.print_exc()

            # ended due to server close/reject or error
            retries += 1
            if self.chunk_index < len(CHUNK_SIZES) - 1:
                self.chunk_index += 1
                self.current_chunk = CHUNK_SIZES[self.chunk_index]
                print(f"📉 Reducing chunk size → {self.current_chunk}")

            backoff = base_backoff * (2 ** (retries - 1))
            print(f"🔁 Reconnecting in {backoff:.1f}s (attempt {retries}/{max_retries})…")
            await asyncio.sleep(backoff)

        self.stop_threads = True
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.chunks_sent / elapsed if elapsed > 0 else 0
            print("\n📊 Stats")
            print(f"  Duration: {elapsed:.1f}s")
            print(f"  Chunks sent: {self.chunks_sent}")
            print(f"  Avg rate: {rate:.1f} chunks/s")
        self.cleanup_audio()
        print("\n👋 Bye")

def main():
    print("🚀 Starting client…")
    try:
        if sys.platform == "win32":
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception:
                pass
        asyncio.run(PTTClient().run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye")
    except Exception as e:
        print(f"\n❌ Unexpected: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()