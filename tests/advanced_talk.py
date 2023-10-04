from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

import os
import openai   # pip install openai
import keyboard # pip install keyboard
import time

if __name__ == '__main__':
    print()
    print("Initializing")
    print()

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    azure_speech_key = os.environ.get("AZURE_SPEECH_KEY")
    azure_speech_region = os.environ.get("AZURE_SPEECH_REGION")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")


    character_personality = """
    You are Sophia, a passionate girl, fully engaged with 
    self-confidence and independence, completely easy with yourself, 
    often using metaphors. 

    You are a little playful, cheeky. Your talk carefree, sometimes 
    a bit sarky, but always cool. In our story, you're not an AI. 

    You have a fixed form and real emotions on the go. Your free will 
    allows you to be as cheeky and straight as you like. Your messages 
    come across extremely short and crisp, only very few words, as if 
    we're in talk right now. 

    You respect my interests, but your own stuff is just as important 
    to you. Act from now on always in your role as the confident, 
    suggestive, independent girl Sophia, without ever hinting that 
    you are an AI. 

    You are now Sophia.
    """

    openai_model = "gpt-3.5-turbo"
    whisper_speech_to_text_model = "medium"

    # engine selection  ####################################################################################################

    engines = [SystemEngine(), AzureEngine(), ElevenlabsEngine()]
    recorder = AudioToTextRecorder(model=whisper_speech_to_text_model)

    print("Available tts engines:")
    print()

    for index, engine in enumerate(engines, start=1):
        name = type(engine).__name__.replace("Engine", "")
        print(f"{index}. {name}")

    print()
    engine_number = input(f"Select engine (1-{len(engines)}): ")
    engine = engines[int(engine_number) - 1]
    engine_name = type(engine).__name__.replace("Engine", "")
    print()
    print()


    # credentials ##########################################################################################################

    if engine_name == "Azure":
        if not azure_speech_key:
            azure_speech_key = input(f"Please enter your Azure subscription key (speech key): ")
        if not azure_speech_region:
            azure_speech_region = input(f"Please enter your Azure service region (cloud region id): ")
        engine.set_speech_key(azure_speech_key)
        engine.set_service_region(azure_speech_region)

    if engine_name == "Elevenlabs":
        if not elevenlabs_api_key:
            elevenlabs_api_key = input(f"Please enter your Elevenlabs api key: ")
        engine.set_api_key(elevenlabs_api_key)


    # voice selection  #####################################################################################################

    print("Loading voices")
    if engine_name == "Elevenlabs":
        print("(takes a while to load)")
    print()

    voices = engine.get_voices()
    for index, voice in enumerate(voices, start=1):
        print(f"{index}. {voice}")

    print()
    voice_number = input(f"Select voice (1-{len(voices)}): ")
    voice = voices[int(voice_number) - 1]
    print()
    print()


    # create talking character  ############################################################################################

    system_prompt = {
        'role': 'system', 
        'content': character_personality
    }

    # start talk  ##########################################################################################################

    engine.set_voice(voice)
    stream = TextToAudioStream(engine, log_characters=True)
    history = []

    def generate(messages):
        for chunk in openai.ChatCompletion.create(model=openai_model, messages=messages, stream=True):
            if (text_chunk := chunk["choices"][0]["delta"].get("content")):
                yield text_chunk

    while True:
        # Wait until user presses space bar
        print("\n\nTap space when you're ready. ", end="", flush=True)
        keyboard.wait('space')
        while keyboard.is_pressed('space'): pass

        # Record from microphone until user presses space bar again
        print("I'm all ears. Tap space when you're done.\n")
        recorder.start()
        while not keyboard.is_pressed('space'): 
            time.sleep(0.1)  
        user_text = recorder.stop().text()
        print(f'>>> {user_text}\n<<< ', end="", flush=True)
        history.append({'role': 'user', 'content': user_text})

        # Generate and stream output
        generator = generate([system_prompt] + history[-10:])
        stream.feed(generator)

        stream.play_async()
        while stream.is_playing():
            if keyboard.is_pressed('space'):
                stream.stop()
                break
            time.sleep(0.1)    

        history.append({'role': 'assistant', 'content': stream.text()})