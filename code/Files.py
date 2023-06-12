import glob
import os
import numpy as np
from tensorflow import keras


class Files:

    def __init__(self, root_folder: str = ""):
        self.root_folder = root_folder

    def files_audio(self):
        # list_audio = []
        # wav_files = glob.glob(os.path.join(self.root_folder, "*.wav"))
        # for file_path in wav_files:
        #     list_audio.append(file_path)
        #     print(file_path)
        # return list_audio

        list_audio = []
        # root_folder = "dataset/noStutter/stutter"
        subfolders = glob.glob(os.path.join(self.root_folder, "**/"), recursive=True)
        for subfolder in subfolders:
            wav_files = glob.glob(os.path.join(subfolder, "*.wav"))
            for file_path in wav_files:
                list_audio.append(file_path)
                print(file_path)
        return list_audio


    def predict_with_model(self, input_vectors):
        print(self.root_folder)
        model = keras.models.load_model(self.root_folder)
        prediction = model.predict(input_vectors)
        binary_predictions = np.round(prediction).flatten()
        return binary_predictions

