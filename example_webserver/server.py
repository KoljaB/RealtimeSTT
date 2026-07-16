WAIT_FOR_START_COMMAND = False

if __name__ == '__main__':
    server = "0.0.0.0"
    port = 5025

    print (f"STT speech to text server")
    print (f"runs on http://{server}:{port}")
    print ()
    print ("starting")
    print ("└─ ... ", end='', flush=True)

    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Style
    import websockets
    import threading
    import colorama
    import asyncio
    import shutil
    import queue
    import json
    import time
    import os

    colorama.init()

    first_chunk = True
    full_sentences = []
    displayed_text = ""
    message_queue = queue.Queue() 
    start_recording_event = threading.Event()
    start_transcription_event = threading.Event()
    connected_clients = set()

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    async def handler(websocket, path):

        print ("\r└─ OK")
        if WAIT_FOR_START_COMMAND:
            print("waiting for start command")
            print ("└─ ... ", end='', flush=True)

        connected_clients.add(websocket)

        try:
            while True:
                async for message in websocket:
                    data = json.loads(message)
                    if data.get("type") == "command" and data.get("content") == "start-recording":
                        print ("\r└─ OK")
                        start_recording_event.set() 

        except json.JSONDecodeError:
            print (Fore.RED + "STT Received an invalid JSON message." + Style.RESET_ALL)
        except websockets.ConnectionClosedError:
            print (Fore.RED + "connection closed unexpectedly by the client" + Style.RESET_ALL)
        except websockets.exceptions.ConnectionClosedOK:
            print("connection closed.")
        finally:

            print("client disconnected")
            connected_clients.remove(websocket)
            print ("waiting for clients")
            print ("└─ ... ", end='', flush=True)


    def add_message_to_queue(type: str, content):
        message = {
            "type": type,
            "content": content
        }
        message_queue.put(message)    

    def fill_cli_line(text):
        columns, _ = shutil.get_terminal_size()
        return text.ljust(columns)[-columns:]

    def text_detected(text):
        global displayed_text, first_chunk

        if text != displayed_text:
            first_chunk = False
            displayed_text = text
            add_message_to_queue("realtime", text)

            message = fill_cli_line(text)

            message ="└─ " + Fore.CYAN + message[:-3] + Style.RESET_ALL
            print(f"\r{message}", end='', flush=True)


    async def broadcast(message_obj):
        if connected_clients:
            for client in connected_clients:
                await client.send(json.dumps(message_obj))

    async def send_handler():
        while True:
            while not message_queue.empty():
                message = message_queue.get()
                await broadcast(message)
            await asyncio.sleep(0.02)

    def recording_started():
        add_message_to_queue("record_start", "")

    def vad_detect_started():
        add_message_to_queue("vad_start", "")

    def wakeword_detect_started():
        add_message_to_queue("wakeword_start", "")

    def transcription_started():
        add_message_to_queue("transcript_start", "")

    recorder_config = {
        'spinner': False,
        'model': 'small.en',
        'language': 'en',
        'silero_sensitivity': 0.01,
        'webrtc_sensitivity': 3,
        'silero_use_onnx': False,
        'post_speech_silence_duration': 1.2,
        'min_length_of_recording': 0.2,
        'min_gap_between_recordings': 0,
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0,
        'realtime_model_type': 'tiny.en',
        'on_realtime_transcription_stabilized': text_detected,
        'on_recording_start' : recording_started,
        'on_vad_detect_start' : vad_detect_started,
        'on_wakeword_detection_start' : wakeword_detect_started,
        'on_transcription_start' : transcription_started,
    }

    recorder = AudioToTextRecorder(**recorder_config)

    def transcriber_thread():
        while True:
            start_transcription_event.wait()
            text = "└─ transcribing ... "
            text = fill_cli_line(text)
            print (f"\r{text}", end='', flush=True)
            sentence = recorder.transcribe()
            print (Style.RESET_ALL + "\r└─ " + Fore.YELLOW + sentence + Style.RESET_ALL)
            add_message_to_queue("full", sentence)
            start_transcription_event.clear()
            if WAIT_FOR_START_COMMAND:
                print("waiting for start command")
                print ("└─ ... ", end='', flush=True)

    def recorder_thread():
        global first_chunk
        while True:
            if not len(connected_clients) > 0:
                time.sleep(0.1)
                continue
            first_chunk = True
            if WAIT_FOR_START_COMMAND:
                start_recording_event.wait() 
            print("waiting for sentence")
            print ("└─ ... ", end='', flush=True)
            recorder.wait_audio()
            start_transcription_event.set()
            start_recording_event.clear()

    threading.Thread(target=recorder_thread, daemon=True).start()
    threading.Thread(target=transcriber_thread, daemon=True).start()

    start_server = websockets.serve(handler, server, port)
    loop = asyncio.get_event_loop()

    print ("\r└─ OK")
    print ("waiting for clients")
    print ("└─ ... ", end='', flush=True)

    loop.run_until_complete(start_server)
    loop.create_task(send_handler())
    loop.run_forever()