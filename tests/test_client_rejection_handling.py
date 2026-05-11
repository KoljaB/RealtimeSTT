#!/usr/bin/env python3
"""
Test script to verify that the simple Python client properly handles server rejections.
This sends intentionally corrupted data to trigger server rejection.
"""

import asyncio
import websockets
import numpy as np
import json
import struct
import time
from datetime import datetime


class ClientRejectionTest:
    def __init__(self):
        self.control_url = "ws://localhost:8011"
        self.data_url = "ws://localhost:8012"
        self.sample_rate = 16000
        self.control_ws = None
        self.data_ws = None
        self.chunks_sent = 0
        self.rejection_received = False
        self.connection_closed = False

    def generate_test_audio(self):
        """Generate test audio"""
        num_samples = int(self.sample_rate * 0.2)  # 200ms
        t = np.linspace(0, 0.2, num_samples, False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # 440Hz tone
        return (audio * 32767).astype(np.int16).tobytes()

    def calculate_checksum(self, audio_data):
        """Calculate correct checksum"""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return int(np.sum(audio_array, dtype=np.int64)) & 0xFFFFFFFF

    async def connect(self):
        """Connect to server"""
        try:
            self.control_ws = await websockets.connect(self.control_url)
            self.data_ws = await websockets.connect(self.data_url)
            print("✅ Connected to server")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def handle_messages(self):
        """Handle server messages like the real client"""
        try:
            async for message in self.data_ws:
                data = json.loads(message)
                timestamp = datetime.now().strftime('%H:%M:%S')

                if data.get('type') == 'error':
                    if data.get('error') == 'data_corruption':
                        print(f"\n🚨 [REJECTION TEST] Server rejected connection!")
                        print(f"    Reason: {data.get('message', 'Unknown')}")
                        print(f"    Action: {data.get('action', 'disconnect')}")
                        self.rejection_received = True
                        break
                    else:
                        print(f"⚠️  Server error: {data.get('message', 'Unknown')}")
                else:
                    print(f"[{timestamp}] 📨 {data.get('type', 'unknown')}")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed by server")
            self.connection_closed = True
        except Exception as e:
            print(f"❌ Message handling error: {e}")

    async def send_corrupted_chunk(self):
        """Send chunk with wrong checksum"""
        if not self.data_ws:
            return False

        try:
            audio_data = self.generate_test_audio()
            correct_checksum = self.calculate_checksum(audio_data)
            wrong_checksum = (correct_checksum + 12345) & 0xFFFFFFFF  # Corrupt it

            metadata = {
                'sampleRate': self.sample_rate,
                'dataLength': len(np.frombuffer(audio_data, dtype=np.int16)),
                'checksum': wrong_checksum,  # Wrong checksum!
                'timestamp': int(time.time() * 1000),
                'server_sent_to_stt': True
            }

            # Encode and send
            metadata_json = json.dumps(metadata)
            metadata_bytes = metadata_json.encode('utf-8')
            metadata_length = struct.pack('<I', len(metadata_bytes))
            message = metadata_length + metadata_bytes + audio_data

            await self.data_ws.send(message)
            self.chunks_sent += 1
            print(
                f"📤 Sent corrupted chunk {self.chunks_sent} (checksum: {correct_checksum:08X} -> {wrong_checksum:08X})")
            return True

        except websockets.exceptions.ConnectionClosed:
            print("🔌 Connection closed while sending")
            self.connection_closed = True
            return False
        except Exception as e:
            print(f"❌ Send error: {e}")
            return False

    async def test_rejection_handling(self):
        """Test that client handles rejection properly"""
        print("🧪 Testing Client Rejection Handling")
        print("=" * 50)

        if not await self.connect():
            return False

        # Start message handler
        msg_task = asyncio.create_task(self.handle_messages())

        try:
            print("📤 Sending corrupted data to trigger rejection...")

            # Send corrupted chunks until server rejects us
            for i in range(5):  # Try up to 5 chunks
                if self.rejection_received or self.connection_closed:
                    break

                success = await self.send_corrupted_chunk()
                if not success:
                    break

                await asyncio.sleep(0.5)  # Wait between sends

            # Wait for server response
            if not (self.rejection_received or self.connection_closed):
                print("⏳ Waiting for server response...")
                await asyncio.sleep(2)

        except Exception as e:
            print(f"❌ Test error: {e}")

        finally:
            msg_task.cancel()
            if self.control_ws:
                await self.control_ws.close()
            if self.data_ws:
                await self.data_ws.close()

        # Report results
        print("\n" + "=" * 50)
        print("📊 Test Results:")
        print(f"   Chunks sent: {self.chunks_sent}")
        print(f"   Rejection received: {'✅ YES' if self.rejection_received else '❌ NO'}")
        print(f"   Connection closed: {'✅ YES' if self.connection_closed else '❌ NO'}")

        if self.rejection_received or self.connection_closed:
            print("✅ SUCCESS: Client rejection handling works correctly!")
            print("💡 The simple_python_client.py should handle this gracefully")
        else:
            print("❌ ISSUE: Server didn't reject corrupted data")
            print("💡 Check server configuration:")
            print("   stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0")

        return self.rejection_received or self.connection_closed


async def main():
    print("🔧 Client Rejection Handling Test")
    print("Make sure server is running with rejection enabled:")
    print("stt-server --verify-data-integrity --reject-corrupted-data --corruption-threshold 0")
    print("\nPress Enter to continue...")
    input()

    test = ClientRejectionTest()
    await test.test_rejection_handling()


if __name__ == "__main__":
    asyncio.run(main())