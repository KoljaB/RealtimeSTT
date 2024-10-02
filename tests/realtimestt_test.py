if __name__ == '__main__':

    EXTENDED_LOGGING = False

    if EXTENDED_LOGGING:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    import os
    import sys
    from RealtimeSTT import AudioToTextRecorder
    from colorama import Fore, Back, Style
    import colorama

    if os.name == "nt" and (3, 8) <= sys.version_info < (3, 99):
        from torchaudio._extension.utils import _init_dll_path
        _init_dll_path()    

    print("Initializing RealtimeSTT test...")

    colorama.init()

    full_sentences = []
    displayed_text = ""
    prev_text = ""
    recorder = None

    end_of_sentence_detection_pause = 0.4
    mid_sentence_detection_pause = 0.7

    def clear_console():
        os.system('clear' if os.name == 'posix' else 'cls')

    def text_detected(text):
        global displayed_text, prev_text
        sentence_end_marks = ['.', '!', '?', '。'] 
        if text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = mid_sentence_detection_pause

        prev_text = text

        sentences_with_style = [
            f"{Fore.YELLOW + sentence + Style.RESET_ALL if i % 2 == 0 else Fore.CYAN + sentence + Style.RESET_ALL} "
            for i, sentence in enumerate(full_sentences)
        ]
        new_text = "".join(sentences_with_style).strip() + " " + text if len(sentences_with_style) > 0 else text

        if new_text != displayed_text:
            displayed_text = new_text
            clear_console()
            print(displayed_text, end="", flush=True)

    def process_text(text):
        recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        full_sentences.append(text)
        prev_text = ""
        text_detected("")

    # Recorder configuration
    recorder_config = {
        'spinner': False,
        'model': 'large-v2',
        'realtime_model_type': 'tiny.en',
        'language': 'en',
        'input_device_index': 1,
        'silero_sensitivity': 0.05,
        'webrtc_sensitivity': 3,
        'post_speech_silence_duration': end_of_sentence_detection_pause,
        'min_length_of_recording': 0,
        'min_gap_between_recordings': 0,                
        'enable_realtime_transcription': True,
        'realtime_processing_pause': 0.1,
        'on_realtime_transcription_update': text_detected,
        'silero_deactivity_detection': True,
        'min_length_of_recording': 0.7,        
        'early_transcription_on_silence': 0.2,
        'beam_size': 5,
        'beam_size_realtime': 1,
        'no_log_file': False,
    }

    if EXTENDED_LOGGING:
        recorder_config['level'] = logging.DEBUG

    recorder = AudioToTextRecorder(**recorder_config)

    clear_console()
    print("Say something...", end="", flush=True)


    try:
        while (True):
            recorder.text(process_text)
    except KeyboardInterrupt:
        print("Exiting application due to keyboard interrupt")
