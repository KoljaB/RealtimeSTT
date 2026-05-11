#!/usr/bin/env python3
"""
Test client that intentionally sends corrupted data to test verification failure detection.
"""

import asyncio
import websockets
import numpy as np
import json
import struct
import time
from datetime import datetime


class CorruptedTestClient:
    def __init__(self,
                 control_url="ws://localhost:8011",
                 data_url="ws://localhost:8012",
                 sample_rate=16000):

        self.control_url = control_url
        self.data_url = data_url
        self.sample_rate = sample_rate

        # State
        self.control_ws = None
        self.data_ws = None
        self.chunks_sent = 0

    def generate_test_audio(self, duration_ms=100):
        """Generate synthetic audio data"""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def calculate_checksum(self, audio_data):
        """Calculate checksum for data verification"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        checksum = int(np.sum(audio_array, dtype=np.int64)) & 0xFFFFFFFF
        return checksum

    async def connect(self):
        """Connect to WebSocket servers"""
        try:
            self.control_ws = await websockets.connect(self.control_url)
            self.data_ws = await websockets.connect(self.data_url)
            print(f"[OK] Connected to servers")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    async def send_corrupted_chunk(self, test_type="wrong_checksum"):
        """Send audio chunk with intentionally corrupted verification data"""
        if not self.data_ws:
            return

        try:
            # Generate original audio
            original_audio = self.generate_test_audio(duration_ms=200)
            actual_checksum = self.calculate_checksum(original_audio)
            actual_length = len(np.frombuffer(original_audio, dtype=np.int16))

            # Prepare metadata with corruption
            if test_type == "wrong_checksum":
                metadata = {
                    'sampleRate': self.sample_rate,
                    'dataLength': actual_length,
                    'checksum': 12345678,  # Wrong checksum
                    'timestamp': int(time.time() * 1000),
                    'server_sent_to_stt': True
                }
                print(f"[TEST] Sending data with WRONG CHECKSUM")
                print(f"       Actual checksum: {actual_checksum:08X}, Sending: {12345678:08X}")
                audio_to_send = original_audio

            elif test_type == "wrong_length":
                metadata = {
                    'sampleRate': self.sample_rate,
                    'dataLength': 9999,  # Wrong length
                    'checksum': actual_checksum,
                    'timestamp': int(time.time() * 1000),
                    'server_sent_to_stt': True
                }
                print(f"[TEST] Sending data with WRONG LENGTH")
                print(f"       Actual length: {actual_length}, Sending: 9999")
                audio_to_send = original_audio

            elif test_type == "corrupted_audio":
                # Actually corrupt the audio data but send correct original checksum
                corrupted_audio = bytearray(original_audio)
                corrupted_audio[100:110] = b'\\x00' * 10  # Corrupt 10 bytes
                audio_to_send = bytes(corrupted_audio)

                metadata = {
                    'sampleRate': self.sample_rate,
                    'dataLength': actual_length,
                    'checksum': actual_checksum,  # Original checksum (should fail)
                    'timestamp': int(time.time() * 1000),
                    'server_sent_to_stt': True
                }
                print(f"[TEST] Sending CORRUPTED AUDIO with original checksum")
                print(f"       Original checksum: {actual_checksum:08X}")

            else:  # Valid data
                metadata = {
                    'sampleRate': self.sample_rate,
                    'dataLength': actual_length,
                    'checksum': actual_checksum,
                    'timestamp': int(time.time() * 1000),
                    'server_sent_to_stt': True
                }
                print(f"[TEST] Sending VALID DATA")
                print(f"       Length: {actual_length}, Checksum: {actual_checksum:08X}")
                audio_to_send = original_audio

            # Encode and send
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_length = struct.pack('<I', len(metadata_bytes))

            message = metadata_length + metadata_bytes + audio_to_send
            await self.data_ws.send(message)

            self.chunks_sent += 1
            print(f"[SEND] Sent test chunk {self.chunks_sent}")

        except Exception as e:
            print(f"[ERROR] Error sending audio chunk: {e}")

    async def handle_data_messages(self):
        """Handle server responses"""
        try:
            async for message in self.data_ws:
                data = json.loads(message)
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                print(f"[{timestamp}] [SERVER] {data}")
        except websockets.exceptions.ConnectionClosed:
            print("[INFO] Connection closed")
        except Exception as e:
            print(f"[ERROR] Error handling messages: {e}")

    async def run_corruption_tests(self):
        """Run various corruption tests"""
        print("[TEST] Corruption Test Client")

        if not await self.connect():
            return False

        # Start message handler
        data_task = asyncio.create_task(self.handle_data_messages())

        test_cases = [
            ("valid", "Valid data (should pass)"),
            ("wrong_checksum", "Wrong checksum (should fail)"),
            ("wrong_length", "Wrong length (should fail)"),
            ("corrupted_audio", "Corrupted audio (should fail)"),
        ]

        try:
            for test_type, description in test_cases:
                print(f"\n--- {description} ---")
                await self.send_corrupted_chunk(test_type)
                await asyncio.sleep(2)  # Wait for server response

            print("\n[OK] All test cases sent. Waiting for final responses...")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"[ERROR] Test failed: {e}")

        finally:
            data_task.cancel()
            if self.control_ws:
                await self.control_ws.close()
            if self.data_ws:
                await self.data_ws.close()

            print(f"\n[SUMMARY] Sent {self.chunks_sent} test chunks")
            print(f"Check server logs for verification results")

        return True


async def main():
    client = CorruptedTestClient()
    await client.run_corruption_tests()


if __name__ == "__main__":
    asyncio.run(main())