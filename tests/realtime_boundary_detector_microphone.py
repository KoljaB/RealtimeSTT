"""
Manual microphone visualizer for RealtimeSpeechBoundaryDetector.

Run from the repository root:
    python tests/realtime_boundary_detector_microphone.py

The status flashes red when the detector commits a likely syllable/speech
boundary. Tune --sensitivity first; use --show-events when you want details.
"""

import argparse
import os
import sys
import time

import pyaudio


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from RealtimeSTT.core.realtime_boundary_detector import RealtimeSpeechBoundaryDetector


ANSI_RESET = "\033[0m"
ANSI_CLEAR_LINE = "\033[K"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_DIM = "\033[2m"


def list_input_devices(audio):
    print("Input devices:")
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if int(info.get("maxInputChannels", 0)) <= 0:
            continue
        default_marker = ""
        try:
            default = audio.get_default_input_device_info()
            if int(default.get("index")) == index:
                default_marker = " (default)"
        except Exception:
            pass
        print(
            "  {index}: {name} [{rate:.0f} Hz]{default}".format(
                index=index,
                name=info.get("name", "unknown"),
                rate=float(info.get("defaultSampleRate", 0)),
                default=default_marker,
            )
        )


def colorize(text, color, enabled):
    if not enabled:
        return text
    return color + text + ANSI_RESET


def db_to_fraction(energy_db):
    low = -70.0
    high = -15.0
    return max(0.0, min(1.0, (energy_db - low) / (high - low)))


def render_bar(value, width):
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "-" * (width - filled)


def render_status_line(result, event_count, flash_until, use_color):
    now = time.time()
    flashing = now < flash_until

    if flashing:
        state = colorize(" SYLLABLE END ", ANSI_RED, use_color)
    elif result.is_vowel_like:
        state = colorize(" vowel        ", ANSI_GREEN, use_color)
    elif result.is_speech:
        state = colorize(" speaking     ", ANSI_GREEN, use_color)
    else:
        state = colorize(" waiting      ", ANSI_YELLOW, use_color)

    bar = render_bar(db_to_fraction(result.current_energy_db), 34)
    colored_bar = colorize(bar, ANSI_GREEN if not flashing else ANSI_RED, use_color)

    return (
        "\r{clear}{state} level [{bar}] "
        "energy={energy:6.1f}dB floor={floor:6.1f}dB voice={voice:0.2f} events={events:04d}"
    ).format(
        clear=ANSI_CLEAR_LINE if use_color else "",
        state=state,
        bar=colored_bar,
        energy=result.current_energy_db,
        floor=result.noise_floor_db,
        voice=result.voicing_score,
        events=event_count,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show realtime speech-boundary detections from the microphone."
    )
    parser.add_argument("--list-devices", action="store_true", help="Print input devices and exit.")
    parser.add_argument("--device-index", type=int, default=None, help="PyAudio input device index.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Input sample rate.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Microphone frames per read.")
    parser.add_argument("--sensitivity", type=float, default=0.6, help="0.0 conservative, 1.0 eager.")
    parser.add_argument("--lookahead-ms", type=float, default=30.0, help="Confirmation lookahead.")
    parser.add_argument(
        "--min-boundary-interval-ms",
        type=float,
        default=110.0,
        help="Minimum time between red flashes.",
    )
    parser.add_argument("--min-voiced-ms", type=float, default=70.0, help="Voiced audio required before a boundary.")
    parser.add_argument("--min-vowel-ms", type=float, default=40.0, help="Vowel-like audio required before a boundary.")
    parser.add_argument("--vowel-margin-db", type=float, default=None, help="Required dB above noise floor for vowel-like frames.")
    parser.add_argument("--min-voicing-score", type=float, default=None, help="Required autocorrelation score for vowel-like frames.")
    parser.add_argument(
        "--max-vowel-zero-crossing-rate",
        type=float,
        default=None,
        help="Maximum zero-crossing rate for vowel-like frames.",
    )
    parser.add_argument("--flash-ms", type=float, default=220.0, help="How long a detection stays red.")
    parser.add_argument("--duration", type=float, default=0.0, help="Optional run duration in seconds.")
    parser.add_argument("--show-events", action="store_true", help="Print one detail line per detection.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    return parser.parse_args()


def open_stream(audio, args):
    return audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=args.sample_rate,
        input=True,
        input_device_index=args.device_index,
        frames_per_buffer=args.chunk_size,
    )


def main():
    args = parse_args()
    use_color = not args.no_color

    audio = pyaudio.PyAudio()
    if args.list_devices:
        try:
            list_input_devices(audio)
        finally:
            audio.terminate()
        return

    detector = RealtimeSpeechBoundaryDetector(
        sample_rate=args.sample_rate,
        sensitivity=args.sensitivity,
        lookahead_ms=args.lookahead_ms,
        min_boundary_interval_ms=args.min_boundary_interval_ms,
        min_voiced_ms=args.min_voiced_ms,
        min_vowel_ms=args.min_vowel_ms,
        vowel_margin_db=args.vowel_margin_db,
        min_voicing_score=args.min_voicing_score,
        max_vowel_zero_crossing_rate=args.max_vowel_zero_crossing_rate,
    )

    stream = None
    flash_until = 0.0
    event_count = 0
    start_time = time.time()

    try:
        stream = open_stream(audio, args)
        print("Speak into the microphone. Press Ctrl+C to stop.")
        print(
            colorize(
                "Tip: if red flashes too often lower --sensitivity; if it misses clear syllables raise it.",
                ANSI_DIM,
                use_color,
            )
        )

        while True:
            if args.duration > 0 and time.time() - start_time >= args.duration:
                break

            chunk = stream.read(args.chunk_size, exception_on_overflow=False)
            result = detector.process_bytes(chunk)

            if result.boundary_detected:
                flash_until = time.time() + args.flash_ms / 1000.0
                event_count += len(result.events)

                if args.show_events:
                    latest = result.latest_event
                    print(
                        "\n{event}".format(
                            event=colorize(
                                "boundary #{count}: t={time:.3f}s score={score:.2f} "
                                "drop={drop:.1f}dB valley={valley:.1f}dB latency={latency:.0f}ms {reason}".format(
                                    count=event_count,
                                    time=latest.boundary_time_seconds,
                                    score=latest.score,
                                    drop=latest.drop_db,
                                    valley=latest.valley_depth_db,
                                    latency=latest.latency_ms,
                                    reason=latest.reason,
                                ),
                                ANSI_RED,
                                use_color,
                            )
                        )
                    )

            sys.stdout.write(render_status_line(result, event_count, flash_until, use_color))
            sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()
        print("\nStopped.")


if __name__ == "__main__":
    main()
