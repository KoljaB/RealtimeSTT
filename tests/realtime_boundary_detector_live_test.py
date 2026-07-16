"""
Long-running live microphone test for the realtime speech-boundary detector.

Run from the repository root:
    python tests/realtime_boundary_detector_live_test.py

This is intentionally not a unittest. It listens until Ctrl+C and flashes red
when the detector thinks a good realtime-transcription boundary was reached.
"""

import os
import runpy
import sys


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    visualizer_path = os.path.join(script_dir, "realtime_boundary_detector_microphone.py")

    if len(sys.argv) == 1:
        sys.argv.extend(["--show-events"])

    runpy.run_path(visualizer_path, run_name="__main__")
