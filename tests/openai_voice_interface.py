import os
import openai
from RealtimeTTS import TextToAudioStream, AzureEngine
from RealtimeSTT import AudioToTextRecorder

if __name__ == '__main__':
    # Initialize OpenAI key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Text-to-Speech Stream Setup
    stream = TextToAudioStream(

        # Alternatives: SystemEngine or ElevenlabsEngine
        AzureEngine(
            os.environ.get("AZURE_SPEECH_KEY"),
            os.environ.get("AZURE_SPEECH_REGION"),
        ),
        log_characters=True
    )

    # Speech-to-Text Recorder Setup
    recorder = AudioToTextRecorder(
        model="medium",
        language="en",
        wake_words="Jarvis",
        spinner=True,
        wake_word_activation_delay=5
    )

    system_prompt_message = {
        'role': 'system',
        'content': 'Answer precise and short with the polite sarcasm of a butler.'
    }

    def generate_response(messages):
        """Generate assistant's response using OpenAI."""
        for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
            text_chunk = chunk["choices"][0]["delta"].get("content")
            if text_chunk:
                yield text_chunk

    history = []

    def main():
        """Main loop for interaction."""
        while True:
            # Capture user input from microphone
            user_text = recorder.text().strip()

            if not user_text:
                continue

            print(f'>>> {user_text}\n<<< ', end="", flush=True)
            history.append({'role': 'user', 'content': user_text})

            # Get assistant response and play it
            assistant_response = generate_response([system_prompt_message] + history[-10:])
            stream.feed(assistant_response).play()

            history.append({'role': 'assistant', 'content': stream.text()})

    if __name__ == "__main__":
        main()
