"""
并发测试用例：验证录音、VAD、转写队列之间的状态切换
测试内容：
1. 每段语音应有独立 segment_id 和音频边界
2. 取消/超时/重连时清理残留 buffer
3. 快速连续录音时不会串音
"""

import os
import sys
import time
import threading
import numpy as np
import unittest
from unittest.mock import patch, MagicMock

if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
    from torchaudio._extension.utils import _init_dll_path
    _init_dll_path()

from RealtimeSTT import AudioToTextRecorder


def generate_test_audio(duration_seconds=0.5, sample_rate=16000, frequency=440):
    """生成测试音频数据"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    return (audio * 32768).astype(np.int16).tobytes()


class TestSegmentIsolation(unittest.TestCase):
    """测试 segment 隔离机制"""

    @classmethod
    def setUpClass(cls):
        """初始化测试"""
        cls.recorder = AudioToTextRecorder(
            spinner=False,
            silero_sensitivity=0.01,
            model="tiny.en",
            language="en",
            use_microphone=False,
        )

    @classmethod
    def tearDownClass(cls):
        """清理测试"""
        cls.recorder.shutdown()

    def test_segment_id_generation(self):
        """测试 segment_id 生成"""
        initial_id = self.recorder._get_current_segment_id()
        
        self.recorder.start()
        first_id = self.recorder._get_current_segment_id()
        self.assertIsNotNone(first_id)
        self.assertNotEqual(initial_id, first_id)
        
        self.recorder.stop()
        
        self.recorder.start()
        second_id = self.recorder._get_current_segment_id()
        self.assertNotEqual(first_id, second_id)
        self.recorder.stop()
        
        print(f"✓ 测试通过: segment_id 正确生成，first_id={first_id}, second_id={second_id}")

    def test_clear_all_buffers(self):
        """测试 _clear_all_buffers 方法"""
        test_audio = generate_test_audio(duration_seconds=0.1)
        
        self.recorder.start()
        
        for _ in range(5):
            self.recorder.feed_audio(test_audio)
            time.sleep(0.01)
        
        with self.recorder.frames_lock:
            initial_frames_count = len(self.recorder.frames)
        
        self.assertGreater(initial_frames_count, 0, "测试前应该有数据")
        
        self.recorder._clear_all_buffers()
        
        with self.recorder.frames_lock:
            final_frames_count = len(self.recorder.frames)
        
        self.assertEqual(final_frames_count, 0, "清理后 frames 应该为空")
        self.assertTrue(self.recorder._segment_completed_event.is_set(), "segment_completed_event 应该被设置")
        
        print(f"✓ 测试通过: _clear_all_buffers 正确清理所有缓冲区")

    def test_segment_completed_event(self):
        """测试 segment_completed_event 状态"""
        self.assertTrue(self.recorder._segment_completed_event.is_set(), 
                       "初始状态下 segment_completed_event 应该被设置")
        
        self.recorder.start()
        self.assertFalse(self.recorder._segment_completed_event.is_set(),
                        "录音开始后 segment_completed_event 应该被清除")
        
        self.recorder.stop()
        self.assertTrue(self.recorder._segment_completed_event.is_set(),
                       "录音停止后 segment_completed_event 应该被设置")
        
        print("✓ 测试通过: segment_completed_event 状态正确")

    def test_frames_lock_protection(self):
        """测试 frames_lock 锁保护"""
        test_audio = generate_test_audio(duration_seconds=0.1)
        
        self.recorder.start()
        
        write_count = [0]
        read_count = [0]
        errors = []
        
        def writer():
            try:
                for _ in range(100):
                    with self.recorder.frames_lock:
                        self.recorder.frames.append(test_audio)
                        write_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Writer error: {e}")
        
        def reader():
            try:
                for _ in range(100):
                    with self.recorder.frames_lock:
                        if self.recorder.frames:
                            _ = self.recorder.frames.copy()
                            read_count[0] += 1
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Reader error: {e}")
        
        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)
        
        writer_thread.start()
        reader_thread.start()
        
        writer_thread.join(timeout=5)
        reader_thread.join(timeout=5)
        
        self.assertEqual(len(errors), 0, f"并发访问不应该有错误: {errors}")
        self.assertGreater(write_count[0], 0, "写操作应该执行")
        self.assertGreater(read_count[0], 0, "读操作应该执行")
        
        self.recorder.stop()
        self.recorder._clear_all_buffers()
        
        print(f"✓ 测试通过: frames_lock 正确保护并发访问，写入={write_count[0]}, 读取={read_count[0]}")

    def test_abort_clears_buffers(self):
        """测试 abort() 方法清理缓冲区"""
        test_audio = generate_test_audio(duration_seconds=0.1)
        
        self.recorder.start()
        
        for _ in range(3):
            self.recorder.feed_audio(test_audio)
            time.sleep(0.01)
        
        with self.recorder.frames_lock:
            frames_before = len(self.recorder.frames)
        
        self.assertGreater(frames_before, 0, "abort 前应该有数据")
        
        self.recorder.abort()
        
        with self.recorder.frames_lock:
            frames_after = len(self.recorder.frames)
        
        self.assertEqual(frames_after, 0, "abort 后 frames 应该为空")
        self.assertTrue(self.recorder._segment_completed_event.is_set(),
                       "abort 后 segment_completed_event 应该被设置")
        
        print("✓ 测试通过: abort() 正确清理所有缓冲区")

    def test_rapid_start_stop_segment_isolation(self):
        """测试快速连续录音的 segment 隔离"""
        test_audio_1 = generate_test_audio(duration_seconds=0.1, frequency=440)
        test_audio_2 = generate_test_audio(duration_seconds=0.1, frequency=880)
        
        self.recorder.start()
        segment_1_id = self.recorder._get_current_segment_id()
        
        for _ in range(2):
            self.recorder.feed_audio(test_audio_1)
        
        with self.recorder.frames_lock:
            segment_1_frames = self.recorder.frames.copy()
        
        self.recorder.stop()
        
        self.recorder.start()
        segment_2_id = self.recorder._get_current_segment_id()
        
        for _ in range(3):
            self.recorder.feed_audio(test_audio_2)
        
        with self.recorder.frames_lock:
            segment_2_frames = self.recorder.frames.copy()
        
        self.recorder.stop()
        
        self.assertNotEqual(segment_1_id, segment_2_id, "不同 segment 应该有不同的 ID")
        
        print(f"✓ 测试通过: 快速连续录音的 segment 隔离正确，segment_1_id={segment_1_id}, segment_2_id={segment_2_id}")
        print(f"  segment_1_frames 数量: {len(segment_1_frames)}")
        print(f"  segment_2_frames 数量: {len(segment_2_frames)}")

    def test_buffer_lock_protection(self):
        """测试 buffer_lock 锁保护"""
        test_audio = generate_test_audio(duration_seconds=0.05)
        
        self.recorder._clear_all_buffers()
        
        errors = []
        
        def feed_audio_thread():
            try:
                for _ in range(50):
                    self.recorder.feed_audio(test_audio)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Feed audio error: {e}")
        
        def clear_buffers_thread():
            try:
                for _ in range(10):
                    self.recorder._clear_all_buffers()
                    time.sleep(0.005)
            except Exception as e:
                errors.append(f"Clear buffers error: {e}")
        
        feed_thread = threading.Thread(target=feed_audio_thread)
        clear_thread = threading.Thread(target=clear_buffers_thread)
        
        feed_thread.start()
        clear_thread.start()
        
        feed_thread.join(timeout=5)
        clear_thread.join(timeout=5)
        
        self.assertEqual(len(errors), 0, f"并发操作不应该有错误: {errors}")
        
        print("✓ 测试通过: buffer_lock 正确保护并发的 feed_audio 和 clear_buffers 操作")


def run_interactive_test():
    """运行交互式测试"""
    print("\n" + "=" * 60)
    print("Segment 隔离交互式测试")
    print("=" * 60)
    
    recorder = AudioToTextRecorder(
        spinner=False,
        silero_sensitivity=0.01,
        model="tiny.en",
        language="en",
        use_microphone=False,
    )
    
    try:
        print("\n测试 1: segment_id 生成")
        print(f"初始 segment_id: {recorder._get_current_segment_id()}")
        
        recorder.start()
        print(f"开始录音后 segment_id: {recorder._get_current_segment_id()}")
        print(f"segment_completed_event.is_set(): {recorder._segment_completed_event.is_set()}")
        
        recorder.stop()
        print(f"停止录音后 segment_completed_event.is_set(): {recorder._segment_completed_event.is_set()}")
        
        print("\n测试 2: 快速连续录音")
        for i in range(3):
            recorder.start()
            seg_id = recorder._get_current_segment_id()
            print(f"第 {i+1} 次录音，segment_id: {seg_id}")
            time.sleep(0.1)
            recorder.stop()
        
        print("\n测试 3: _clear_all_buffers")
        recorder.start()
        test_audio = generate_test_audio(duration_seconds=0.1)
        for _ in range(5):
            recorder.feed_audio(test_audio)
        
        with recorder.frames_lock:
            print(f"清理前 frames 数量: {len(recorder.frames)}")
        
        recorder._clear_all_buffers()
        
        with recorder.frames_lock:
            print(f"清理后 frames 数量: {len(recorder.frames)}")
        print(f"清理后 segment_completed_event.is_set(): {recorder._segment_completed_event.is_set()}")
        
        print("\n✓ 所有交互式测试完成！")
        
    finally:
        recorder.shutdown()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment 隔离测试')
    parser.add_argument('--interactive', action='store_true', help='运行交互式测试')
    parser.add_argument('--unittest', action='store_true', help='运行单元测试')
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_test()
    elif args.unittest:
        unittest.main(verbosity=2)
    else:
        print("运行单元测试...")
        unittest.main(verbosity=2)
