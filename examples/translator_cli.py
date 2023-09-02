from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, AzureEngine
import os, openai

openai.api_key = os.environ.get("OPENAI_API_KEY")
engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), service_region="germanywestcentral") # exchange "germanywestcentral" with your azure service region

stream = TextToAudioStream(engine)
recorder = AudioToTextRecorder(model="medium")

languages = [["english", "AshleyNeural"],
             ["german", "AmalaNeural"],
             ["french", "DeniseNeural"],
             ["spanish", "EstrellaNeural"],
             ["portuguese", "FernandaNeural"],
             ["italian", "FabiolaNeural"]]

def generate_charstream(messages):
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            yield text_chunk

for index, language in enumerate(languages, start=1):
    print(f"{index}. {language[0]}")

language_number = input("Select translation language (1-6): ")
language_info = languages[int(language_number) - 1]
engine.set_voice(language_info[1])
system_prompt = {'role': 'system', 'content': f'Translate the given text to {language_info[0]}. Output only the translated text.'}

while True:
    print("\nSay something!")
    user_text = recorder.text()
    print(f"Input text: {user_text}")

    print("Translation: ", end="", flush=True)
    user_message = {'role': 'user', 'content': user_text}
    translation_stream = generate_charstream([system_prompt, user_message])

    stream.feed(translation_stream)
    recorder.noise_detection(False)
    stream.play(log_characters=True)
    recorder.noise_detection(True)