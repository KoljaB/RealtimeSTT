from colorama import Fore, Style
import websockets
import colorama
import keyboard
import asyncio
import json
import os

colorama.init()

SEND_START_COMMAND = False
HOST = 'localhost:5025'
URI = f'ws://{HOST}'
RECONNECT_DELAY = 5  

full_sentences = []

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def update_displayed_text(text = ""):
    sentences_with_style = [
        f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
        for i, sentence in enumerate(full_sentences)
    ]
    text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text
    clear_console()
    print("CLIENT retrieved text:")
    print()
    print(text)

async def send_start_recording(websocket):
    command = {
        "type": "command",
        "content": "start-recording"
    }
    await websocket.send(json.dumps(command))

async def test_client():
    while True:
        try:
            async with websockets.connect(URI, ping_interval=None) as websocket:

                if SEND_START_COMMAND:
                    # New: Check for space bar press and send start-recording message
                    async def check_space_keypress():
                        while True:
                            if keyboard.is_pressed('space'):
                                print ("Space bar pressed. Sending start-recording message to server.")
                                await send_start_recording(websocket)
                                await asyncio.sleep(1) 
                            await asyncio.sleep(0.02)
                    
                    # Start a task to monitor the space keypress
                    print ("Press space bar to start recording.")
                    asyncio.create_task(check_space_keypress())
                
                while True:
                    message = await websocket.recv()
                    message_obj = json.loads(message)
                    
                    if message_obj["type"] == "realtime":
                        clear_console()
                        print (message_obj["content"])
                    elif message_obj["type"] == "full":
                        clear_console()
                        colored_message = Fore.YELLOW + message_obj["content"] + Style.RESET_ALL
                        print (colored_message)
                        print ()
                        if SEND_START_COMMAND:
                            print ("Press space bar to start recording.")
                        full_sentences.append(message_obj["content"])
                    elif message_obj["type"] == "record_start":
                        print ("recording started.")
                    elif message_obj["type"] == "vad_start":
                        print ("vad started.")
                    elif message_obj["type"] == "wakeword_start":
                        print ("wakeword started.")
                    elif message_obj["type"] == "transcript_start":
                        print ("transcript started.")

                    else:
                        print (f"Unknown message: {message_obj}")
                    
        except websockets.ConnectionClosed:
            print("Connection with server closed. Reconnecting in", RECONNECT_DELAY, "seconds...")
            await asyncio.sleep(RECONNECT_DELAY)
        except KeyboardInterrupt:
            print("Gracefully shutting down the client.")
            break
        except Exception as e:
            print(f"An error occurred: {e}. Reconnecting in", RECONNECT_DELAY, "seconds...")
            await asyncio.sleep(RECONNECT_DELAY)    

asyncio.run(test_client())