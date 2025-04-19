"""
pip install realtimestt realtimetts[edge]
"""

# Set this to False to start by waiting for a wake word first
# Set this to True to start directly in voice activity mode
START_IN_VOICE_ACTIVITY_MODE = False

if __name__ == '__main__':
    import os
    import openai
    from RealtimeTTS import TextToAudioStream, EdgeEngine
    from RealtimeSTT import AudioToTextRecorder

    # Text-to-Speech Stream Setup (EdgeEngine)
    engine = EdgeEngine(rate=0, pitch=0, volume=0)
    engine.set_voice("en-US-SoniaNeural")
    stream = TextToAudioStream(
        engine,
        log_characters=True
    )

    # Speech-to-Text Recorder Setup
    recorder = AudioToTextRecorder(
        model="medium",
        language="en",
        wake_words="Jarvis",
        spinner=True,
        wake_word_activation_delay=5 if START_IN_VOICE_ACTIVITY_MODE else 0,
    )

    system_prompt_message = {
        'role': 'system',
        'content': 'Answer precise and short with the polite sarcasm of a butler.'
    }

    def generate_response(messages):
        """Generate assistant's response using OpenAI."""
        response_stream = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )

        for chunk in response_stream:
            text_chunk = chunk.choices[0].delta.content
            if text_chunk:
                yield text_chunk

    history = []

    try:
        # Main loop for interaction
        while True:
            if START_IN_VOICE_ACTIVITY_MODE:
                print("Please speak...")
            else:
                print('Say "Jarvis" then speak...')

            user_text = recorder.text().strip()

            # If not starting in voice activity mode, set the delay after the first interaction
            if not START_IN_VOICE_ACTIVITY_MODE:
                recorder.wake_word_activation_delay = 5

            print(f"Transcribed: {user_text}")

            if not user_text:
                continue

            print(f'>>> {user_text}\n<<< ', end="", flush=True)
            history.append({'role': 'user', 'content': user_text})

            # Get assistant response and play it
            assistant_response = generate_response([system_prompt_message] + history[-10:])
            stream.feed(assistant_response).play()

            history.append({'role': 'assistant', 'content': stream.text()})
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Shutting down...")
        recorder.shutdown()
