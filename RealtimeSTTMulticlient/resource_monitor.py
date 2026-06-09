#!/usr/bin/env python3
"""
Resource monitoring script for AudioToTextRecorder instances
Tests memory, CPU, and GPU usage for the specific models used in stt_server.py
"""

import psutil
import time
import threading
import json
import sys
import os
from datetime import datetime
import traceback

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPUtil not available - GPU monitoring disabled")

try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIMESTT_AVAILABLE = True
except ImportError:
    REALTIMESTT_AVAILABLE = False
    print("RealtimeSTT not available - cannot test actual recorder instances")

class ResourceMonitor:
    def __init__(self, monitor_interval=1.0):
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.monitor_thread = None
        self.baseline_memory = None
        self.baseline_cpu = None
        self.baseline_gpu = None
        self.measurements = []
        
    def get_current_resources(self):
        """Get current system resource usage"""
        process = psutil.Process()
        
        # Memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU usage
        cpu_percent = process.cpu_percent()
        
        # GPU usage (if available)
        gpu_info = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_info = {
                        'memory_used_mb': gpu.memoryUsed,
                        'memory_total_mb': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'utilization_percent': gpu.load * 100
                    }
            except Exception as e:
                gpu_info = {'error': str(e)}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'gpu': gpu_info
        }
    
    def set_baseline(self):
        """Set baseline resource usage"""
        resources = self.get_current_resources()
        self.baseline_memory = resources['memory_mb']
        self.baseline_cpu = resources['cpu_percent']
        self.baseline_gpu = resources['gpu']
        print(f"Baseline set - Memory: {self.baseline_memory:.1f}MB, CPU: {self.baseline_cpu:.1f}%")
        if self.baseline_gpu:
            print(f"Baseline GPU - Memory: {self.baseline_gpu.get('memory_used_mb', 'N/A')}MB, Util: {self.baseline_gpu.get('utilization_percent', 'N/A')}%")
    
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Resource monitoring started...")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                resources = self.get_current_resources()
                self.measurements.append(resources)
                time.sleep(self.monitor_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
    
    def get_resource_diff(self):
        """Calculate resource usage difference from baseline"""
        if not self.measurements or not self.baseline_memory:
            return None
        
        current = self.measurements[-1]
        memory_diff = current['memory_mb'] - self.baseline_memory
        cpu_diff = current['cpu_percent'] - self.baseline_cpu
        
        gpu_diff = None
        if current['gpu'] and self.baseline_gpu and 'memory_used_mb' in current['gpu'] and 'memory_used_mb' in self.baseline_gpu:
            gpu_diff = {
                'memory_mb_diff': current['gpu']['memory_used_mb'] - self.baseline_gpu['memory_used_mb'],
                'utilization_diff': current['gpu']['utilization_percent'] - self.baseline_gpu.get('utilization_percent', 0)
            }
        
        return {
            'memory_mb_diff': memory_diff,
            'cpu_percent_diff': cpu_diff,
            'gpu_diff': gpu_diff,
            'current': current,
            'baseline': {
                'memory_mb': self.baseline_memory,
                'cpu_percent': self.baseline_cpu,
                'gpu': self.baseline_gpu
            }
        }
    
    def generate_report(self):
        """Generate resource usage report"""
        if not self.measurements:
            return "No measurements available"
        
        # Calculate statistics
        memory_values = [m['memory_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        
        memory_diff = max(memory_values) - self.baseline_memory if self.baseline_memory else 0
        avg_memory_diff = sum(memory_values) / len(memory_values) - self.baseline_memory if self.baseline_memory else 0
        
        report = f"""
Resource Usage Report
====================
Duration: {len(self.measurements)} measurements over {len(self.measurements) * self.monitor_interval:.1f} seconds

Memory Usage:
- Baseline: {self.baseline_memory:.1f}MB
- Peak: {max(memory_values):.1f}MB
- Average: {sum(memory_values) / len(memory_values):.1f}MB
- Peak Increase: {memory_diff:.1f}MB
- Average Increase: {avg_memory_diff:.1f}MB

CPU Usage:
- Peak: {max(cpu_values):.1f}%
- Average: {sum(cpu_values) / len(cpu_values):.1f}%
"""
        
        if GPU_AVAILABLE and any(m['gpu'] and 'memory_used_mb' in m['gpu'] for m in self.measurements):
            gpu_memory_values = [m['gpu']['memory_used_mb'] for m in self.measurements if m['gpu'] and 'memory_used_mb' in m['gpu']]
            gpu_util_values = [m['gpu']['utilization_percent'] for m in self.measurements if m['gpu'] and 'utilization_percent' in m['gpu']]
            
            if gpu_memory_values:
                gpu_memory_diff = max(gpu_memory_values) - (self.baseline_gpu['memory_used_mb'] if self.baseline_gpu and 'memory_used_mb' in self.baseline_gpu else 0)
                report += f"""
GPU Usage:
- Peak Memory: {max(gpu_memory_values)}MB
- Average Memory: {sum(gpu_memory_values) / len(gpu_memory_values):.1f}MB
- Memory Increase: {gpu_memory_diff}MB
- Peak Utilization: {max(gpu_util_values):.1f}%
- Average Utilization: {sum(gpu_util_values) / len(gpu_util_values):.1f}%
"""
        
        return report

def test_single_recorder_instance():
    """Test resource usage of a single AudioToTextRecorder instance"""
    if not REALTIMESTT_AVAILABLE:
        print("RealtimeSTT not available - skipping recorder test")
        return
    
    print("\n" + "="*60)
    print("Testing Single AudioToTextRecorder Instance")
    print("="*60)
    
    monitor = ResourceMonitor()
    monitor.set_baseline()
    monitor.start_monitoring()
    
    try:
        # Create recorder with your app's exact configuration
        config = {
            'model': 'small.en',  # Your main model
            'realtime_model_type': 'base.en',  # Your realtime model
            'language': 'en',
            'batch_size': 16,
            'realtime_batch_size': 16,
            'use_microphone': False,  # Don't use actual microphone
            'spinner': False,
            'no_log_file': True,
            'level': 'WARNING'
        }
        
        print("Creating AudioToTextRecorder instance...")
        recorder = AudioToTextRecorder(**config)
        
        # Let it initialize fully
        time.sleep(5)
        print("Instance created, monitoring for 10 seconds...")
        time.sleep(10)
        
        # Test feeding some audio data
        print("Testing audio processing...")
        # Create dummy audio data (16-bit PCM, 16kHz, 1 second)
        import numpy as np
        dummy_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        
        for i in range(5):
            recorder.feed_audio(dummy_audio)
            time.sleep(0.5)
        
        print("Audio processing test completed, monitoring for 5 more seconds...")
        time.sleep(5)
        
        # Cleanup
        recorder.shutdown()
        print("Recorder shutdown completed")
        time.sleep(2)
        
    except Exception as e:
        print(f"Error during recorder test: {e}")
        traceback.print_exc()
    finally:
        monitor.stop_monitoring()
        print(monitor.generate_report())

def test_multiple_recorder_instances(num_instances=3):
    """Test resource usage of multiple AudioToTextRecorder instances"""
    if not REALTIMESTT_AVAILABLE:
        print("RealtimeSTT not available - skipping multiple recorder test")
        return
    
    print("\n" + "="*60)
    print(f"Testing {num_instances} AudioToTextRecorder Instances")
    print("="*60)
    
    monitor = ResourceMonitor()
    monitor.set_baseline()
    monitor.start_monitoring()
    
    recorders = []
    
    try:
        config = {
            'model': 'small.en',
            'realtime_model_type': 'base.en',
            'language': 'en',
            'batch_size': 16,
            'realtime_batch_size': 16,
            'use_microphone': False,
            'spinner': False,
            'no_log_file': True,
            'level': 'WARNING'
        }
        
        # Create instances one by one
        for i in range(num_instances):
            print(f"Creating recorder instance {i+1}/{num_instances}...")
            recorder = AudioToTextRecorder(**config)
            recorders.append(recorder)
            time.sleep(2)  # Allow each instance to initialize
            
            # Show resource usage after each instance
            diff = monitor.get_resource_diff()
            if diff:
                print(f"  After {i+1} instances - Memory: +{diff['memory_mb_diff']:.1f}MB")
        
        print(f"All {num_instances} instances created, monitoring for 10 seconds...")
        time.sleep(10)
        
        # Test concurrent audio processing
        print("Testing concurrent audio processing...")
        import numpy as np
        dummy_audio = np.random.randint(-1000, 1000, 16000, dtype=np.int16).tobytes()
        
        for i in range(3):
            for j, recorder in enumerate(recorders):
                recorder.feed_audio(dummy_audio)
            time.sleep(1)
        
        print("Concurrent processing test completed")
        time.sleep(5)
        
    except Exception as e:
        print(f"Error during multiple recorder test: {e}")
        traceback.print_exc()
    finally:
        # Cleanup all recorders
        print("Shutting down all recorders...")
        for i, recorder in enumerate(recorders):
            try:
                recorder.shutdown()
                print(f"Recorder {i+1} shutdown")
            except Exception as e:
                print(f"Error shutting down recorder {i+1}: {e}")
        
        monitor.stop_monitoring()
        print(monitor.generate_report())

def main():
    print("AudioToTextRecorder Resource Monitor")
    print("====================================")
    print(f"RealtimeSTT Available: {REALTIMESTT_AVAILABLE}")
    print(f"GPU Monitoring Available: {GPU_AVAILABLE}")
    print()
    
    # Test single instance
    test_single_recorder_instance()
    
    # Wait a bit between tests
    time.sleep(5)
    
    # Test multiple instances
    test_multiple_recorder_instances(3)

if __name__ == "__main__":
    main()