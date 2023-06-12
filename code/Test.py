import numpy as np
from tensorflow import keras
from Files import Files
from Method import Method


class Test:
    def __init__(self, stuttering_folder: str = "dataset/test/stutter",
                 no_stuttering_folder: str = "dataset/test/noStutter", classified: str = "classification_model.h5"):
        self.stuttering_folder = stuttering_folder
        self.no_stuttering_folder = no_stuttering_folder
        self.classified = classified
        self.test_model()

    def test_model(self):
        model = keras.models.load_model(self.classified)
        audio_no_stutter = Files(self.no_stuttering_folder).files_audio()
        print(audio_no_stutter)
        audio_stutter = Files(self.stuttering_folder).files_audio()
        test_vectors = []
        test_labels = []
        test_vectors_to = []
        for i in range(len(audio_no_stutter)):
            concatenated_vector_get = Method(audio_no_stutter[i]).get_concatenated_vector_get()
            print(concatenated_vector_get)
            prediction = model.predict(concatenated_vector_get)
            binary_predictions = np.round(prediction).flatten()
            print("binary_predictions[0] ", binary_predictions[0])
            print("test_labels 0 ")

            test_vectors_to.append(binary_predictions[0])
            test_vectors.append(concatenated_vector_get)
            test_labels.append(0)
        for i in range(len(audio_stutter)):
            concatenated_vector_get = Method(audio_stutter[i]).get_concatenated_vector_get()
            test_vectors.append(concatenated_vector_get)
            prediction = model.predict(concatenated_vector_get)
            binary_predictions = np.round(prediction).flatten()
            print("binary_predictions[0] ", binary_predictions[0])
            print("test_labels 1 ")
            test_vectors_to.append(binary_predictions[0])
            test_labels.append(1)

        precision = self.calculate_precision(test_labels, test_vectors_to)
        recall = self.calculate_recall(test_labels, test_vectors_to)
        f1_score = self.calculate_f1_score(test_labels, test_vectors_to)
        accuracy = self.calculate_accuracy(test_labels, test_vectors_to)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1_score)
        print("Accuracy:", accuracy)

    def calculate_precision(self, labels, predicted_labels):
        true_positives = 0
        predicted_positives = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1:
                predicted_positives += 1
                if labels[i] == 1:
                    true_positives += 1

        precision = true_positives / predicted_positives if predicted_positives != 0 else 0
        return precision

    def calculate_recall(self,labels, predicted_labels):
        true_positives = 0
        actual_positives = 0
        for i in range(len(predicted_labels)):
            if labels[i] == 1:
                actual_positives += 1
                if predicted_labels[i] == 1:
                    true_positives += 1
        recall = true_positives / actual_positives if actual_positives != 0 else 0
        return recall

    def calculate_f1_score(self,labels, predicted_labels):
        precision = self.calculate_precision(labels, predicted_labels)
        recall = self.calculate_recall(labels, predicted_labels)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score

    def calculate_accuracy(self,labels, predicted_labels):
        correct_predictions = 0
        total_predictions = len(labels)
        for i in range(len(predicted_labels)):
            if labels[i] == predicted_labels[i]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy