import tkinter as tk
from tkinter import *
import os
import sounddevice as sd
from tkinter import filedialog
import soundfile as sf
from DetectStuttering import DetectStuttering

class User:
    def __init__(self):
        window = tk.Tk()
        window.geometry("800x400")
        window.title("Voice Analysis")
        left_frame = tk.Frame(window)
        left_frame.pack(side=tk.TOP, padx=10)
        text_audio_label = tk.Label(left_frame, text=" Target Text : ")
        text_audio_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.text_entry = tk.Text(left_frame, height=1, width=15)
        self.text_entry.grid(row=0, column=2, columnspan=2, rowspan=2,
                        padx=5, pady=10)
        duration_label = tk.Label(left_frame, text=" Voice Recoder : ")
        duration_label.grid(row=2, column=0, padx=5, pady=10, sticky=tk.W)
        self.file_name_entry = tk.Entry(left_frame)
        self.file_name_entry.grid(row=2, column=2, columnspan=2, rowspan=2,
                             padx=5, pady=10)
        b = Button(left_frame, text="Start", command=self.voice_rec)
        b.grid(row=2, column=4, columnspan=2, rowspan=2,
               padx=5, pady=10)
        duration_label_or = tk.Label(left_frame, text=" OR ")
        duration_label_or.grid(row=2, column=6, columnspan=2, rowspan=2,
                               padx=5, pady=10)
        self.browse_button = tk.Button(left_frame, text="Browse Audio", command=self.browse_audio)
        self.browse_button.grid(row=2, column=8, columnspan=2, rowspan=2,
                           padx=5, pady=10)
        self.loading_label = tk.Label(left_frame, text="", font=("Arial", 12, "bold"))
        self.loading_label.grid(row=6, column=2, columnspan=2, rowspan=2,
                           padx=5, pady=10)
        send_button = tk.Button(left_frame, text="Send", width=20, command=self.send_text)
        send_button.grid(row=10, column=2, columnspan=2, rowspan=2,
                         padx=5, pady=10)
        line_frame = tk.Frame(window, height=1, bg="black")
        line_frame.pack(fill=tk.X)
        self.file_path_audio = None
        right_frame = tk.Frame(window)
        right_frame.pack(side=tk.BOTTOM, padx=10)
        result_title = tk.Label(right_frame, text="Results", font=("Arial", 14, "bold"))
        result_title.grid(row=0, column=0, columnspan=2, rowspan=2,
                          padx=5, pady=10)
        screen_width = window.winfo_screenwidth() / 2
        self.result_text = tk.Label(right_frame, text="", font=("Arial", 10), wraplength=screen_width)
        self.result_text.grid(row=2, column=0, columnspan=2, rowspan=2,
                         padx=5, pady=10)

        window.mainloop()

    def start_loading(self):
        self.loading_label.config(text="Loading...")
        self.browse_button.config(state=tk.DISABLED)

    def stop_loading(self):
        self.loading_label.config(text="")
        self.browse_button.config(state=tk.NORMAL)

    def browse_audio(self):
        file_path = tk.filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            self.file_path_audio = file_path
            file_name = os.path.basename(file_path)
            self.file_name_entry.delete(0, tk.END)
            self.file_name_entry.insert(tk.END, file_name)

    def voice_rec(self):
        fs = 16000
        duration = 5
        myrecording = sd.rec(int(duration * fs),
                             samplerate=fs, channels=2)
        sd.wait()
        sf.write('audio_file.wav', myrecording, fs)
        self.file_path_audio = "audio_file.wav"
        self.file_name_entry.delete(0, tk.END)
        self.file_name_entry.insert(tk.END, self.file_path_audio)

    def send_text(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        print("Text:", text)
        print("file_path_audio ", self.file_path_audio)
        self.start_loading()
        full_text = DetectStuttering(self.file_path_audio, text).predict_megamgem()
        print(full_text)
        self.result_text.config(text=full_text)
        self.stop_loading()