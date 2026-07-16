from RealtimeSTT import AudioToTextRecorderClient

# ANSI escape codes for terminal control
CLEAR_LINE = "\033[K"      # Clear from cursor to end of line
RESET_CURSOR = "\r"        # Move cursor to the beginning of the line
GREEN_TEXT = "\033[92m"    # Set text color to green
RESET_COLOR = "\033[0m"    # Reset text color to default

def print_realtime_text(text):
    print(f"{RESET_CURSOR}{CLEAR_LINE}{GREEN_TEXT}👄 {text}{RESET_COLOR}", end="", flush=True)

# Initialize the audio recorder with the real-time transcription callback
recorder = AudioToTextRecorderClient(on_realtime_transcription_update=print_realtime_text)

# Print the speaking prompt
print("👄 ", end="", flush=True)

try:
    while True:
        # Fetch finalized transcription text, if available
        if text := recorder.text():
            # Display the finalized transcription
            print(f"{RESET_CURSOR}{CLEAR_LINE}✍️ {text}\n👄 ", end="", flush=True)
except KeyboardInterrupt:
    # Handle graceful shutdown on Ctrl+C
    print(f"{RESET_CURSOR}{CLEAR_LINE}", end="", flush=True)
    recorder.shutdown()