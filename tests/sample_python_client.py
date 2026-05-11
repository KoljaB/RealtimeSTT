#!/usr/bin/env python3
"""
Simple Python WebSocket client for RealtimeSTT server with data integrity verification.
Just run it directly - no command line arguments needed!

Usage:
    python simple_python_client.py

The client will:
- Connect to localhost STT server (ports 8011/8012)
- Use your default microphone
- Enable data integrity verification
- Record until you press Ctrl+C
"""

import asyncio
import websockets
import pyaudio
import numpy as np
import json
import struct
import time
import threading
from datetime import datetime


class SimpleRealtimeSTTClient:
    def __init__(self):
        # Fixed configuration - no arguments needed
        self.control_url = "ws://localhost:8011"
        self.data_url = "ws://localhost:8012"
        self.sample_rate = 16000  # Match server expectation
        self.chunk_size = 4096  # Larger chunks for better performance
        self.verify_data_integrity = True  # Always enabled

        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # State
        self.running = False
        self.control_ws = None
        self.data_ws = None

        # Statistics
        self.chunks_sent = 0
        self.start_time = None
        self.current_transcription = ""

    def find_microphone(self):
        """Find the best available microphone"""
        print("🎤 Looking for microphone...")

        # Try to find a good input device
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                try:
                    # Test if this device works
                    test_stream = self.audio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024
                    )
                    test_stream.close()
                    print(f"✓ Using microphone: {info['name']}")
                    return i
                except:
                    continue

        print("⚠️  No working microphone found, using system default")
        return None

    def setup_audio(self):
        """Initialize audio recording"""
        try:
            device_index = self.find_microphone()

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            return True
        except Exception as e:
            print(f"❌ Error setting up audio: {e}")
            return False

    def cleanup_audio(self):
        """Clean up audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def calculate_checksum(self, audio_data):
        """Calculate checksum for data verification"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        checksum = int(np.sum(audio_array, dtype=np.int64)) & 0xFFFFFFFF
        return checksum

    async def connect(self):
        """Connect to WebSocket servers"""
        try:
            print("🔗 Connecting to STT server...")
            self.control_ws = await websockets.connect(self.control_url)
            self.data_ws = await websockets.connect(self.data_url)
            print("✅ Connected to STT server!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Make sure the STT server is running:")
            print("   stt-server --model tiny --control_port 8011 --data_port 8012 --verify-data-integrity")
            return False

    async def handle_data_messages(self):
        """Handle incoming transcription results and server messages"""
        try:
            async for message in self.data_ws:
                data = json.loads(message)
                timestamp = datetime.now().strftime('%H:%M:%S')

                # Handle server rejection/error messages
                if data.get('type') == 'error':
                    if data.get('error') == 'data_corruption':
                        print(f"\n\n🚨 [REJECTED] Server rejected connection due to data corruption!")
                        print(f"    Reason: {data.get('message', 'Unknown corruption error')}")
                        print(f"    Action: {data.get('action', 'disconnect')}")
                        print(f"\n💡 This indicates a problem with audio data transmission.")
                        print(f"    Possible causes:")
                        print(f"    - Network issues corrupting audio packets")
                        print(f"    - Microphone driver problems")
                        print(f"    - System audio processing issues")
                        print(f"\n🔧 Try:")
                        print(f"    - Restart the client")
                        print(f"    - Check your network connection")
                        print(f"    - Try a different microphone")

                        # Stop processing to allow graceful shutdown
                        self.running = False
                        break
                    else:
                        print(f"\n⚠️  [ERROR] Server error: {data.get('message', 'Unknown error')}")

                elif data.get('type') == 'realtime':
                    text = data.get('text', '').strip()
                    if text:
                        # Check if this is a continuation of current transcription
                        if text.startswith(self.current_transcription):
                            # Update current line
                            self.current_transcription = text
                            print(f"\r[{timestamp}] 🎤 {text}", end='', flush=True)
                        else:
                            # New transcription
                            if self.current_transcription:
                                print()  # New line
                            self.current_transcription = text
                            print(f"[{timestamp}] 🎤 {text}", end='', flush=True)

                elif data.get('type') == 'fullSentence':
                    text = data.get('text', '')
                    print(f"\n[{timestamp}] ✅ Final: {text}")
                    self.current_transcription = ""

                elif data.get('type') == 'recording_start':
                    print(f"\n[{timestamp}] 🔴 Recording started")

                elif data.get('type') == 'recording_stop':
                    print(f"\n[{timestamp}] ⏹️  Recording stopped")

                # Silently handle other message types (they're normal operation)
                elif data.get('type') in ['vad_detect_start', 'vad_detect_stop', 'transcription_start',
                                          'start_turn_detection', 'stop_turn_detection', 'wakeword_detected',
                                          'wakeword_detection_start', 'wakeword_detection_end']:
                    # These are normal server messages, don't spam the user
                    pass

                else:
                    # Only log truly unknown message types
                    if data.get('type') and data.get('type') not in ['realtime', 'fullSentence']:
                        print(f"\n[{timestamp}] 📨 Unknown: {data.get('type')}")

        except websockets.exceptions.ConnectionClosed:
            print("\n🔌 Server connection closed")
        except Exception as e:
            print(f"\n❌ Error handling messages: {e}")

    def send_audio_chunk(self, audio_data):
        """Send audio chunk with verification data"""
        if not self.data_ws:
            return

        try:
            # Prepare metadata with verification data
            metadata = {
                'sampleRate': self.sample_rate,
                'dataLength': len(np.frombuffer(audio_data, dtype=np.int16)),
                'checksum': self.calculate_checksum(audio_data),
                'timestamp': int(time.time() * 1000),
                'server_sent_to_stt': True  # Enable verification

            }

            # Encode metadata
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_length = struct.pack('<I', len(metadata_bytes))

            # Combine message
            message = metadata_length + metadata_bytes + audio_data

            # Schedule sending from the audio thread
            if hasattr(self, 'loop') and self.loop:
                asyncio.run_coroutine_threadsafe(self.data_ws.send(message), self.loop)

            self.chunks_sent += 1

        except websockets.exceptions.ConnectionClosed:
            print(f"\n🔌 Connection closed while sending audio")
            self.running = False
        except Exception as e:
            print(f"\n❌ Error sending audio: {e}")
            # If we get repeated errors, stop trying to send
            if "connection" in str(e).lower() or "closed" in str(e).lower():
                self.running = False

    def audio_thread(self):
        """Audio recording thread"""
        print("🎙️  Recording audio...")

        while self.running:
            try:
                # Read audio data
                audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                # Send to server with verification
                self.send_audio_chunk(audio_data)

            except Exception as e:
                print(f"\n❌ Audio thread error: {e}")
                break

    async def run(self):
        """Main run method"""
        print("=" * 50)
        print("🎯 Simple RealtimeSTT Python Client")
        print("   Data integrity verification: ✅ Enabled")
        print("   Sample rate: 16000 Hz")
        print("   Server: localhost:8011/8012")
        print("   Server rejection: Depends on server config")
        print("=" * 50)
        print("💡 Server config examples:")
        print("   # Strict (reject on first corruption):")
        print("   stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0")
        print("   # Tolerant (allow 3 failures):")
        print("   stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 3")

        # Store event loop reference for audio thread
        self.loop = asyncio.get_event_loop()

        # Setup audio
        if not self.setup_audio():
            return False

        # Connect to server
        if not await self.connect():
            self.cleanup_audio()
            return False

        # Start recording
        self.running = True
        self.start_time = time.time()

        # Start audio recording thread
        audio_thread = threading.Thread(target=self.audio_thread)
        audio_thread.daemon = True
        audio_thread.start()

        # Start message handler
        data_task = asyncio.create_task(self.handle_data_messages())

        try:
            print("\n🎤 Recording started! Say something...")
            print("   Press Ctrl+C to stop")
            print("-" * 30)

            # Keep running until interrupted
            while self.running:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping...")

        finally:
            # Cleanup
            self.running = False
            audio_thread.join(timeout=1.0)

            # Cancel message handler
            data_task.cancel()

            # Close connections
            if self.control_ws:
                await self.control_ws.close()
            if self.data_ws:
                await self.data_ws.close()

            self.cleanup_audio()

            # Show statistics
            if self.start_time:
                elapsed = time.time() - self.start_time
                rate = self.chunks_sent / elapsed if elapsed > 0 else 0
                print(f"\n📊 Session Stats:")
                print(f"   Duration: {elapsed:.1f} seconds")
                print(f"   Audio chunks sent: {self.chunks_sent}")
                print(f"   Average rate: {rate:.1f} chunks/sec")
                print(f"   Data verification: ✅ Enabled")

        print("\n👋 Thanks for using RealtimeSTT!")
        return True


def main():
    """Simple main function - no arguments needed!"""
    print("🚀 Starting Simple RealtimeSTT Client...")

    client = SimpleRealtimeSTTClient()

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Make sure the STT server is running with:")
        print("   stt-server --model tiny --control_port 8011 --data_port 8012 --verify-data-integrity")


if __name__ == "__main__":
    main()