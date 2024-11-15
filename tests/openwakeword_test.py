if __name__ == '__main__':
    print("Starting...")
    from RealtimeSTT import AudioToTextRecorder

    detected = False

    say_wakeword_str = "Listening for wakeword 'samantha'."

    def on_wakeword_detected():
        global detected
        detected = True

    def on_recording_stop():
        print ("Transcribing...")
    
    def on_wakeword_timeout():
        global detected
        if not detected:
            print(f"Timeout. {say_wakeword_str}")

        detected = False

    def on_wakeword_detection_start():
        print(f"\n{say_wakeword_str}")

    def on_recording_start():
        print ("Recording...")

    def on_vad_detect_start():
        print()
        print()

    def text_detected(text):
        print(f">> {text}")

    with AudioToTextRecorder(
        spinner=False,
        model="large-v2",
        language="en", 
        wakeword_backend="oww",
        wake_words_sensitivity=0.35,
        # openwakeword_model_paths="model_wake_word1.onnx,model_wake_word2.onnx",
        openwakeword_model_paths="suh_man_tuh.onnx,suh_mahn_thuh.onnx", # load these test models from https://huggingface.co/KoljaB/SamanthaOpenwakeword/tree/main and save in tests folder
        on_wakeword_detected=on_wakeword_detected,
        on_recording_start=on_recording_start,
        on_recording_stop=on_recording_stop,
        on_wakeword_timeout=on_wakeword_timeout,
        on_wakeword_detection_start=on_wakeword_detection_start,
        on_vad_detect_start=on_vad_detect_start,
        wake_word_buffer_duration=1,
        ) as recorder:

        while (True):                
            recorder.text(text_detected)
