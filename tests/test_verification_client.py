#!/usr/bin/env python3
"""
Test client that sends synthetic audio data to verify the data integrity verification system.
This doesn't require a real microphone.
"""

import asyncio
import websockets
import numpy as np
import json
import struct
import time
from datetime import datetime


class TestVerificationClient:
    def __init__(self,
                 control_url="ws://localhost:8011",
                 data_url="ws://localhost:8012",
                 sample_rate=16000,  # Match server expectation
                 verify_data_integrity=True):

        self.control_url = control_url
        self.data_url = data_url
        self.sample_rate = sample_rate
        self.verify_data_integrity = verify_data_integrity

        # State
        self.control_ws = None
        self.data_ws = None

        # Statistics
        self.chunks_sent = 0
        self.verifications_passed = 0
        self.verifications_failed = 0

    def generate_test_audio(self, duration_ms=100):
        """Generate synthetic audio data for testing"""
        num_samples = int(self.sample_rate * duration_ms / 1000)

        # Generate a simple sine wave
        t = np.linspace(0, duration_ms / 1000, num_samples, False)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% volume

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def calculate_checksum(self, audio_data):
        """Calculate checksum for data verification"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        checksum = int(np.sum(audio_array, dtype=np.int64)) & 0xFFFFFFFF
        return checksum

    async def connect(self):
        """Connect to both WebSocket servers"""
        try:
            self.control_ws = await websockets.connect(self.control_url)
            print(f"[OK] Connected to control server: {self.control_url}")

            self.data_ws = await websockets.connect(self.data_url)
            print(f"[OK] Connected to data server: {self.data_url}")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            return False

    async def handle_data_messages(self):
        """Handle incoming data messages"""
        try:
            async for message in self.data_ws:
                data = json.loads(message)
                timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                if data.get('type') == 'realtime':
                    text = data.get('text', '').strip()
                    if text:
                        print(f"[{timestamp}] [REALTIME] {text}")

                elif data.get('type') == 'fullSentence':
                    text = data.get('text', '')
                    print(f"[{timestamp}] [FINAL] {text}")

                elif data.get('type') == 'recording_start':
                    print(f"[{timestamp}] [RECORD] Recording started")

                elif data.get('type') == 'recording_stop':
                    print(f"[{timestamp}] [RECORD] Recording stopped")

                else:
                    print(f"[{timestamp}] [INFO] {data}")

        except websockets.exceptions.ConnectionClosed:
            print("Data connection closed")
        except Exception as e:
            print(f"Error handling data messages: {e}")

    async def send_audio_chunk(self, audio_data):
        """Send audio chunk with verification metadata"""
        if not self.data_ws:
            return

        try:
            # Prepare metadata
            metadata = {
                'sampleRate': self.sample_rate,
            }

            # Add verification data
            if self.verify_data_integrity:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                checksum = self.calculate_checksum(audio_data)
                metadata.update({
                    'dataLength': len(audio_array),
                    'checksum': checksum,
                    'timestamp': int(time.time() * 1000),
                    'server_sent_to_stt': True
                })

            # Encode metadata
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_length = struct.pack('<I', len(metadata_bytes))

            # Combine message
            message = metadata_length + metadata_bytes + audio_data

            # Send via WebSocket
            await self.data_ws.send(message)

            self.chunks_sent += 1

            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            if self.verify_data_integrity:
                print(
                    f"[{timestamp}] [SEND] Sent chunk {self.chunks_sent}: {len(audio_array)} samples, checksum: {checksum:08X}")
            else:
                print(f"[{timestamp}] [SEND] Sent chunk {self.chunks_sent}: {len(audio_array)} samples")

        except Exception as e:
            print(f"[ERROR] Error sending audio chunk: {e}")

    async def run_test(self, num_chunks=10, chunk_interval=0.5):
        """Run test sending synthetic audio chunks"""
        print(f"[TEST] Test Client - Data Integrity Verification")
        print(f"   Verification: {'Enabled' if self.verify_data_integrity else 'Disabled'}")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Chunks to send: {num_chunks}")
        print(f"   Interval: {chunk_interval}s")

        if not await self.connect():
            return False

        # Start message handler
        data_task = asyncio.create_task(self.handle_data_messages())

        try:
            print(f"\n[START] Starting test...")

            for i in range(num_chunks):
                # Generate test audio
                audio_data = self.generate_test_audio(duration_ms=200)  # 200ms chunks

                # Send chunk
                await self.send_audio_chunk(audio_data)

                # Wait before next chunk
                if i < num_chunks - 1:  # Don't wait after last chunk
                    await asyncio.sleep(chunk_interval)

            print(f"\n[OK] Sent all {num_chunks} chunks. Waiting for server responses...")

            # Wait a bit for server to process
            await asyncio.sleep(3.0)

        except Exception as e:
            print(f"[ERROR] Test failed: {e}")

        finally:
            # Cleanup
            data_task.cancel()

            if self.control_ws:
                await self.control_ws.close()
            if self.data_ws:
                await self.data_ws.close()

            print(f"\n[SUMMARY] Test Summary:")
            print(f"   Chunks sent: {self.chunks_sent}")
            if self.verify_data_integrity:
                print(f"   Server should have verified data integrity for each chunk")
                print(f"   Check server logs for verification results")

        return True


async def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test client for data integrity verification')
    parser.add_argument('--control-url', default='ws://localhost:8011',
                        help='Control WebSocket URL')
    parser.add_argument('--data-url', default='ws://localhost:8012',
                        help='Data WebSocket URL')
    parser.add_argument('--chunks', type=int, default=10,
                        help='Number of audio chunks to send')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Interval between chunks in seconds')
    parser.add_argument('--no-verify', action='store_true',
                        help='Disable data integrity verification')

    args = parser.parse_args()

    client = TestVerificationClient(
        control_url=args.control_url,
        data_url=args.data_url,
        verify_data_integrity=not args.no_verify
    )

    await client.run_test(num_chunks=args.chunks, chunk_interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())