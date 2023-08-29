import RealtimeSTT 

def recording_started():
    print("Speak now...")

def recording_finished():
    print("Speech end detected... transcribing...")

recorder = RealtimeSTT.AudioToTextRecorder(model="small.en", language="en", wake_words="jarvis", on_recording_started=recording_started, on_recording_finished=recording_finished)

print('Say "Jarvis" then speak.')
print(recorder.text())