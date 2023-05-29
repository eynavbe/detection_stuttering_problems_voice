import librosa
from transformers import BertTokenizer, TFAutoModel
import glob
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import tkinter as tk
from tkinter import *
import sounddevice as sd
from tkinter import filedialog
import soundfile as sf
from torch.nn import CTCLoss
import torch
import torchaudio
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFAutoModel.from_pretrained('bert-base-uncased')

def files_audio(root_folder):
    list_audio = []
    wav_files = glob.glob(os.path.join(root_folder, "*.wav"))
    for file_path in wav_files:
        list_audio.append(file_path)
        print(file_path)
    return list_audio


def convert_audio_to_matrix(audio_file):
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000)
    audio_input, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_input = resampler(audio_input)
        sample_rate = 16000
    audio_features = feature_extractor(
        audio_input,
        sampling_rate=sample_rate,
        padding=True,
        return_tensors="pt"
    )
    matrix = audio_features.input_values.squeeze(0)
    return matrix


def generate_embedding_vector(transcript):
    inputs = tokenizer.encode_plus(transcript, add_special_tokens=True, return_tensors='tf')
    outputs = model(inputs['input_ids'])
    embedding_vector = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding_vector


def vector_concatenation(bert_vector, wav2vec_matrix):
    m = wav2vec_matrix.shape[0]
    n = bert_vector.shape[0]
    if wav2vec_matrix.ndim == 1:
        wav2vec_matrix = np.expand_dims(wav2vec_matrix, axis=1)
    bert_vector = np.tile(bert_vector, (m, 1))
    concatenated_vector = np.concatenate((wav2vec_matrix, bert_vector), axis=1)
    return concatenated_vector


def binary_classifier_training(ambidig_vectors, labels):
    concatenated_vectors = np.concatenate(ambidig_vectors, axis=0)
    concatenated_labels = np.repeat(labels, [v.shape[0] for v in ambidig_vectors], axis=0)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=769))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(concatenated_vectors, concatenated_labels, epochs=10, batch_size=32)
    model.save('classification_model.h5')
    print('Model weights saved successfully.')


def transcribe_audio(audio_path):
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    input_values = tokenizer(waveform[0], return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def target_transcript(audio_file):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    input_values = processor(waveform[0], return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    return predicted_sentences[0]


def transcribe_detect_stuttering(audio_path, proper_transcript):
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    print(proper_transcript)
    if proper_transcript == "":
        proper_transcript = target_transcript(audio_path)
        proper_transcript = proper_transcript.upper()
    else:
        proper_transcript = proper_transcript.upper()
    input_audio, _ = librosa.load(audio_path, sr=16000)
    input_features = tokenizer(input_audio, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_features).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    stuttering_indices = []
    for i, (char, proper_char) in enumerate(zip(transcription, proper_transcript)):
        if char != proper_char:
            stuttering_indices.append(i)
    stuttering_segments = []
    if stuttering_indices:
        current_segment = [stuttering_indices[0]]
        for i in range(1, len(stuttering_indices)):
            if stuttering_indices[i] - stuttering_indices[i - 1] > 1:
                stuttering_segments.append(current_segment)
                current_segment = [stuttering_indices[i]]
            else:
                current_segment.append(stuttering_indices[i])
        stuttering_segments.append(current_segment)
    proper_transcript_length = len(proper_transcript)
    total_stuttered_chars = 0
    stuttered_chars = []
    for segment in stuttering_segments:
        stuttered_chars.append(transcription[segment[0]:segment[-1] + 1])
        total_stuttered_chars = total_stuttered_chars + len(transcription[segment[0]:segment[-1] + 1])
    stuttering_percentage = (total_stuttered_chars / proper_transcript_length) * 100
    stuttering_percentage = round(stuttering_percentage)
    full_text = ""
    full_text += "Stuttering Percentage: " + str(stuttering_percentage) + "%" + "\n"
    full_text += "Stuttered Chars: ["
    for i in range(len(stuttered_chars)):
        if i != len(stuttered_chars) - 1:
            full_text += " " + stuttered_chars[i] + ", "
        else:
            full_text += " " + stuttered_chars[i] + "] " + "\n"
    full_text += "Transcription: " + transcription + "\n"
    full_text += "Transcript without stuttering: " + proper_transcript + "\n"
    return full_text


def matrix_to_one_vector(matrix):
    mean_vector = torch.mean(matrix, dim=0)
    return mean_vector


def stuttering_seconds(audio_file, target_transcription):
    if target_transcription == "":
        target_transcription = target_transcript(audio_file)
        target_transcription = target_transcription.upper()
    else:
        target_transcription = target_transcription.upper()
    waveform, sample_rate = torchaudio.load(audio_file)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    input_values = processor(waveform[0], return_tensors="pt").input_values  # Modified this line
    with torch.no_grad():
        logits = model(input_values).logits
    tokens = processor.tokenizer(target_transcription, return_tensors="pt", padding=True, truncation=True)
    target_ids = tokens.input_ids
    log_probs = torch.log_softmax(logits, dim=-1)
    input_lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long)
    target_lengths = torch.tensor([target_ids.shape[1]], dtype=torch.long)
    ctc_loss = CTCLoss(blank=model.config.pad_token_id)
    loss = ctc_loss(log_probs.transpose(0, 1), targets=target_ids, input_lengths=input_lengths,
                    target_lengths=target_lengths)
    model_name = 'facebook/wav2vec2-large-960h-lv60-self'
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    input_audio, sample_rate = librosa.load(audio_file, sr=16000)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        input_audio = resampler(input_audio)
    input_features = tokenizer(input_audio.squeeze(), return_tensors='pt').input_values
    with torch.no_grad():
        logits = model(input_features).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    blank_token_id = tokenizer.pad_token_id
    stuttering_seconds = []
    prev_token = blank_token_id
    frame_shift = 0.02
    audio_duration = len(input_audio) / sample_rate
    for frame_idx, token_id in enumerate(predicted_ids[0]):
        if token_id != blank_token_id and token_id != prev_token:
            start_frame = frame_idx
            end_frame = frame_idx + token_id.item() - 1
            start_second = min(start_frame * frame_shift, audio_duration)
            end_second = min(end_frame * frame_shift, audio_duration)
            if end_second - start_second > 0.4:
                stuttering_seconds.append((round(start_second, 2), round(end_second, 2)))
        prev_token = token_id
    return round(loss.item(), 2), stuttering_seconds


def method(audio_file):
    sound_file_path = audio_file
    characters_get = transcribe_audio(sound_file_path)
    convert_audio_to_matrix_get = convert_audio_to_matrix(sound_file_path)
    matrix_to_one_vector_get = matrix_to_one_vector(convert_audio_to_matrix_get)
    generate_ambidig_vector_get = generate_embedding_vector(characters_get)
    concatenated_vector_get = vector_concatenation(generate_ambidig_vector_get, matrix_to_one_vector_get)
    return concatenated_vector_get


def initial_training_model():
    audio_no_stutter = files_audio("dataset/train/noStutter")
    audio_stutter = files_audio("dataset/train/stutter")
    list_concatenated_vector_get = []
    labels = []
    for i in range(len(audio_no_stutter)):
        concatenated_vector_get = method(audio_no_stutter[i])
        list_concatenated_vector_get.append(concatenated_vector_get)
        labels.append(0)
    for i in range(len(audio_stutter)):
        concatenated_vector_get = method(audio_stutter[i])
        list_concatenated_vector_get.append(concatenated_vector_get)
        labels.append(1)
    binary_classifier_training(list_concatenated_vector_get, labels)


def calculate_precision(labels, predicted_labels):
    true_positives = 0
    predicted_positives = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 1:
            predicted_positives+=1
            if labels[i] == 1:
                true_positives += 1

    precision = true_positives / predicted_positives if predicted_positives != 0 else 0
    return precision


def calculate_recall(labels, predicted_labels):
    true_positives = 0
    actual_positives = 0
    for i in range(len(predicted_labels)):
        if labels[i] == 1:
            actual_positives +=1
            if predicted_labels[i] == 1:
                true_positives += 1
    recall = true_positives / actual_positives if actual_positives != 0 else 0
    return recall


def calculate_f1_score(labels, predicted_labels):
    precision = calculate_precision(labels, predicted_labels)
    recall = calculate_recall(labels, predicted_labels)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return f1_score


def calculate_accuracy(labels, predicted_labels):
    correct_predictions = 0
    total_predictions = len(labels)
    for i in range(len(predicted_labels)):
        if labels[i] == predicted_labels[i]:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy



def test_model():
    import numpy as np
    from tensorflow import keras
    model = keras.models.load_model('classification_model.h5')
    audio_no_stutter = files_audio("dataset/test/noStutter")
    print(audio_no_stutter)
    audio_stutter = files_audio("dataset/test/stutter")
    test_vectors = []
    test_labels = []
    test_vectors_to = []
    for i in range(len(audio_no_stutter)):
        concatenated_vector_get = method(audio_no_stutter[i])
        prediction = model.predict(concatenated_vector_get)
        binary_predictions = np.round(prediction).flatten()
        print("binary_predictions[0] ",binary_predictions[0])
        print("test_labels 0 ")

        test_vectors_to.append(binary_predictions[0])
        test_vectors.append(concatenated_vector_get)
        test_labels.append(0)
    for i in range(len(audio_stutter)):
        concatenated_vector_get = method(audio_stutter[i])
        test_vectors.append(concatenated_vector_get)
        prediction = model.predict(concatenated_vector_get)
        binary_predictions = np.round(prediction).flatten()
        print("binary_predictions[0] ", binary_predictions[0])
        print("test_labels 1 ")
        test_vectors_to.append(binary_predictions[0])
        test_labels.append(1)
        # Compute evaluation metrics for each test set

    precision = calculate_precision(test_labels, test_vectors_to)
    recall = calculate_recall(test_labels, test_vectors_to)
    f1_score = calculate_f1_score(test_labels, test_vectors_to)
    accuracy = calculate_accuracy(test_labels, test_vectors_to)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
    print("Accuracy:", accuracy)


def predict_with_model(input_vectors):
    import numpy as np
    from tensorflow import keras
    model = keras.models.load_model('classification_model.h5')
    prediction = model.predict(input_vectors)
    binary_predictions = np.round(prediction).flatten()
    return binary_predictions


def predict_megamgem(sound_file_path, text):
    concatenated_vector_get = method(sound_file_path)
    prediction = predict_with_model(concatenated_vector_get)
    print("prediction ", prediction[0])
    if (prediction[0] == 1):
        print("stutterer")
        proper_transcript = text.upper()
        loss, stuttering_seconds_get = stuttering_seconds(sound_file_path, proper_transcript)
        stuttering_seconds_str = ', '.join(str(sec) for sec in stuttering_seconds_get)
        full_text = transcribe_detect_stuttering(sound_file_path, proper_transcript)
        full_text = "stutterer" + "\n\n" +\
                    "stuttering seconds: " + stuttering_seconds_str + "\n" + full_text
        full_text = full_text.lower()
        return full_text
    else:
        return "no stutterer"


if __name__ == "__main__":
    # initial_training_model()
    # test_model()

    def start_loading():
        loading_label.config(text="Loading...")
        browse_button.config(state=tk.DISABLED)


    def stop_loading():
        loading_label.config(text="")
        browse_button.config(state=tk.NORMAL)


    def browse_audio():
        file_path = tk.filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if file_path:
            global file_path_audio
            file_path_audio = file_path
            file_name = os.path.basename(file_path)
            file_name_entry.delete(0, tk.END)
            file_name_entry.insert(tk.END, file_name)


    def voice_rec():
        fs = 16000
        duration = 5
        myrecording = sd.rec(int(duration * fs),
                             samplerate=fs, channels=2)
        sd.wait()
        sf.write('audio_file.wav', myrecording, fs)
        global file_path_audio
        file_path_audio = "audio_file.wav"
        file_name_entry.delete(0, tk.END)
        file_name_entry.insert(tk.END, file_path_audio)


    def send_text():
        text = text_entry.get("1.0", tk.END).strip()
        print("Text:", text)
        print("file_path_audio ", file_path_audio)
        start_loading()
        full_text = predict_megamgem(file_path_audio, text)
        print(full_text)
        result_text.config(text=full_text)
        stop_loading()


    window = tk.Tk()
    window.geometry("800x400")
    window.title("Voice Analysis")
    left_frame = tk.Frame(window)
    left_frame.pack(side=tk.TOP, padx=10)
    text_audio_label = tk.Label(left_frame, text=" Target Text : ")
    text_audio_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    text_entry = tk.Text(left_frame, height=1, width=15)
    text_entry.grid(row=0, column=2, columnspan=2, rowspan=2,
                    padx=5, pady=10)
    duration_label = tk.Label(left_frame, text=" Voice Recoder : ")
    duration_label.grid(row=2, column=0, padx=5, pady=10, sticky=tk.W)
    file_name_entry = tk.Entry(left_frame)
    file_name_entry.grid(row=2, column=2, columnspan=2, rowspan=2,
                         padx=5, pady=10)
    b = Button(left_frame, text="Start", command=voice_rec)
    b.grid(row=2, column=4, columnspan=2, rowspan=2,
           padx=5, pady=10)
    duration_label_or = tk.Label(left_frame, text=" OR ")
    duration_label_or.grid(row=2, column=6, columnspan=2, rowspan=2,
                           padx=5, pady=10)
    browse_button = tk.Button(left_frame, text="Browse Audio", command=browse_audio)
    browse_button.grid(row=2, column=8, columnspan=2, rowspan=2,
                       padx=5, pady=10)
    loading_label = tk.Label(left_frame, text="", font=("Arial", 12, "bold"))
    loading_label.grid(row=6, column=2, columnspan=2, rowspan=2,
                       padx=5, pady=10)
    send_button = tk.Button(left_frame, text="Send", width=20, command=send_text)
    send_button.grid(row=10, column=2, columnspan=2, rowspan=2,
                     padx=5, pady=10)
    line_frame = tk.Frame(window, height=1, bg="black")
    line_frame.pack(fill=tk.X)
    file_path_audio = None
    right_frame = tk.Frame(window)
    right_frame.pack(side=tk.BOTTOM, padx=10)
    result_title = tk.Label(right_frame, text="Results", font=("Arial", 14, "bold"))
    result_title.grid(row=0, column=0, columnspan=2, rowspan=2,
                      padx=5, pady=10)
    screen_width = window.winfo_screenwidth() / 2
    result_text = tk.Label(right_frame, text="", font=("Arial", 10), wraplength=screen_width)
    result_text.grid(row=2, column=0, columnspan=2, rowspan=2,
                     padx=5, pady=10)

    window.mainloop()
