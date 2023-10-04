import os
import openai
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, AzureEngine

if __name__ == '__main__':
    # Setup OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Text-to-Speech Stream Setup (alternative engines: SystemEngine or ElevenlabsEngine)
    engine = AzureEngine( 
        os.environ.get("AZURE_SPEECH_KEY"),
        os.environ.get("AZURE_SPEECH_REGION")
    )
    stream = TextToAudioStream(engine, log_characters=True)

    # Speech-to-Text Recorder Setup
    recorder = AudioToTextRecorder(
        model="medium",
    )

    # Supported languages and their voices
    languages = [
        ["english", "AshleyNeural"],
        ["german", "AmalaNeural"],
        ["french", "DeniseNeural"],
        ["spanish", "EstrellaNeural"],
        ["portuguese", "FernandaNeural"],
        ["italian", "FabiolaNeural"]
    ]

    def generate_response(messages):
        """Generate assistant's response using OpenAI."""
        for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
            text_chunk = chunk["choices"][0]["delta"].get("content")
            if text_chunk:
                yield text_chunk
                
    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def select_language():
        """Display language options and get user's choice."""
        for index, language in enumerate(languages, start=1):
            print(f"{index}. {language[0]}")
        language_number = input("Select language to translate to (1-6): ")
        return languages[int(language_number) - 1]

    def main():
        """Main translation loop."""
        clear_console()
        language_info = select_language()
        engine.set_voice(language_info[1])

        system_prompt_message = {
            'role': 'system',
            'content': f'Translate the given text to {language_info[0]}. Output only the translated text.'
        }

        while True:
            print("\nSay something!")

            # Capture user input from microphone
            user_text = recorder.text()
            print(f"Input text: {user_text}")

            user_message = {'role': 'user', 'content': user_text}

            # Get assistant response and play it
            translation_stream = generate_response([system_prompt_message, user_message])
            print("Translation: ", end="", flush=True)
            stream.feed(translation_stream)
            stream.play()

    main()