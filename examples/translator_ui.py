import openai
import tkinter as tk
from tkinter import Scrollbar, Text
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, AzureEngine
import threading
import os
import logging
import time
import math

openai.api_key = os.environ.get("OPENAI_API_KEY")
engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "germanywestcentral")


system_prompt = {
    'role': 'system', 
    'content': 'Translate the given text to english. Output only the english text. Input: "Hallo, wie geht es dir?" Output: "Hello, how are you?"'
}

stream = TextToAudioStream(engine)
output = ""


# Global lists to store our canvases and their rectangles
canvases = []
level_rects = []

def generate(messages):
    global output
    output = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            output += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

def update_status(text):
    status_label.config(text=text)

def map_value(value):
    return min(max((math.log(value * 4, 1.5) - 10) / 10, 0.0), 1.0)

def update_levels(levels):
    for level, canvas, rect in zip(levels, canvases, level_rects):
        y = canvas_height - (canvas_height * level)
        canvas.coords(rect, 15, y, 65, canvas_height)

def generate_translation(user_text):
    user_message = {'role': 'user', 'content': user_text}
    generator = generate([system_prompt] + [user_message])
    stream.feed(generator)
    recorder.long_term_noise_calculation = False
    stream.play()
    recorder.long_term_noise_calculation = True

    start_recording()

def wait_for_recording_finished():
    user_text = recorder.text()
    input_text.config(state=tk.NORMAL)
    input_text.delete(1.0, tk.END)
    input_text.insert(tk.END, user_text)
    input_text.config(state=tk.DISABLED)

    threading.Thread(target=generate_translation, args=(user_text,)).start()
    root.after(0, update_status, "translating")

def start_recording():
    threading.Thread(target=wait_for_recording_finished).start()

recorder = AudioToTextRecorder(
    model="medium",
    language="de", 
)

def update_output_ui():
    update_levels([map_value(recorder.level_short_term), map_value(recorder.level_long_term), map_value(recorder.level_floor), map_value(recorder.level_peak), recorder.voice_deactivity_probability])
    update_status(recorder.state if recorder.long_term_noise_calculation else "translating")
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, output)
    output_text.config(state=tk.DISABLED)

def update_output():
    root.after(0,update_output_ui)
    root.after(100, update_output) 
    

root = tk.Tk()
root.title("Translator")
root.geometry("600x900")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

status_label = tk.Label(frame, text="Initializing...")
status_label.pack(pady=10)

input_text_label = tk.Label(frame, text="Input")
input_text_label.pack(pady=10)

input_text = Text(frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
input_text.pack(pady=10)

output_text_label = tk.Label(frame, text="Output")
output_text_label.pack(pady=10)

output_text = Text(frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
output_text.pack(pady=10)

canvas_width = 80
canvas_height = 200

canvas_frame = tk.Frame(frame) 
canvas_frame.pack(pady=20)

for _ in range(5):  
    canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack(side=tk.LEFT, padx=5)
    canvases.append(canvas)

    rect = canvas.create_rectangle(15, canvas_height, 65, canvas_height, fill="blue")
    level_rects.append(rect)

root.after(500, start_recording)
root.after(500, update_output) 
root.mainloop()