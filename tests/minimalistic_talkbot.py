import RealtimeSTT, RealtimeTTS
import openai, os

if __name__ == '__main__':
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    character_prompt = 'Answer precise and short with the polite sarcasm of a butler.'
    stream = RealtimeTTS.TextToAudioStream(RealtimeTTS.AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")), log_characters=True)
    recorder = RealtimeSTT.AudioToTextRecorder(model="medium")

    def generate(messages):
        for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
            if (text_chunk := chunk["choices"][0]["delta"].get("content")): yield text_chunk

    history = []
    while True:
        print("\n\nSpeak when ready")
        print(f'>>> {(user_text := recorder.text())}\n<<< ', end="", flush=True)
        history.append({'role': 'user', 'content': user_text})
        assistant_response = generate([{ 'role': 'system',  'content': character_prompt}] + history[-10:])
        stream.feed(assistant_response).play()
        history.append({'role': 'assistant', 'content': stream.text()})