from RealtimeSTT import AudioToTextRecorder

def recording_started():
    print(" >> recording started... ", end="", flush=True)

def recording_finished():
    print("recording finished...")

recorder = AudioToTextRecorder(language="de", on_recording_started=recording_started, on_recording_finished=recording_finished)


# usage 1:
# automatic detection of speech start and end, waits for text to be returned
print ("Say something...")
print (f'TEXT: "{recorder.text()}"')
print()


# usage 2:
# manual trigger of speech start and end
print("Tap space when you're ready.")
import keyboard, time
keyboard.wait('space')
while keyboard.is_pressed('space'): time.sleep(0.1)

recorder.start()

print("tap space when you're done... ", end="", flush=True)
while not keyboard.is_pressed('space'): time.sleep(0.1)

print (f'TEXT: "{recorder.stop().text()}"')