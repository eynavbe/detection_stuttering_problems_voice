from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from keras.activations import sigmoid, relu
from keras.layers import Dense
from Files import Files
from Method import Method


class Train:
    def __init__(self, stuttering_folder: str = "dataset/train/stutter", no_stuttering_folder: str = "dataset/train/noStutter"):
        self.stuttering_folder = stuttering_folder
        self.no_stuttering_folder = no_stuttering_folder
        self.initial_training_model()

    def initial_training_model(self):
        audio_no_stutter = Files(self.no_stuttering_folder).files_audio()
        print(audio_no_stutter)
        audio_stutter = Files(self.stuttering_folder).files_audio()
        list_concatenated_vector_get = []
        labels = []
        for i in range(0*int(len(audio_no_stutter)/7),1*int(len(audio_no_stutter)/7)):
            print(audio_no_stutter[i])
            concatenated_vector_get = Method(audio_no_stutter[i]).get_concatenated_vector_get()
            list_concatenated_vector_get.append(concatenated_vector_get)
            labels.append(0)
        for i in range(0*int(len(audio_stutter)/7),1*int(len(audio_stutter)/7)):
            print(audio_stutter[i])
            concatenated_vector_get = Method(audio_stutter[i]).get_concatenated_vector_get()
            list_concatenated_vector_get.append(concatenated_vector_get)
            labels.append(1)
        self.binary_classifier_training(list_concatenated_vector_get, labels)



    def binary_classifier_training(self, ambidig_vectors, labels):
        concatenated_vectors = np.concatenate(ambidig_vectors, axis=0)
        concatenated_labels = np.repeat(labels, [v.shape[0] for v in ambidig_vectors], axis=0)
        model = Sequential()
        model.add(Dense(64, activation=relu, input_dim=769))
        model.add(Dense(64, activation=relu))
        model.add(Dense(64, activation=relu))
        model.add(Dense(1, activation=sigmoid))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(concatenated_vectors, concatenated_labels, epochs=10, batch_size=32)
        model.save('classification_model.h5')
        print('Model weights saved successfully.')
