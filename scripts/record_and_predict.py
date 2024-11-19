import tensorflow as tf
import os
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
import sounddevice as sd
import scipy.io.wavfile as wav

base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/audio_processing')
dataset_dir = os.path.join(base_dir, 'dataset')
models_dir = os.path.join(base_dir, 'model')
model_path = os.path.join(models_dir, 'speech_to_text.keras')
logs_dir = os.path.join(base_dir, 'logs')

#custom layer for channel attention mechanism
@tf.keras.utils.register_keras_serializable()
class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self, ratio = 8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()

    def build(self, input_shape):
        self.fc1 = layers.Dense(units = input_shape[-1] // self.ratio, activation = 'relu')
        self.fc2 = layers.Dense(units = input_shape[-1], activation = 'sigmoid')

    def call(self, inputs):
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))
        out = avg_out + max_out
        out = tf.expand_dims(tf.expand_dims(out, axis = 1), axis = 1)
        return inputs * out

#function to load and preprocess audio to fixed length and sampling rate
def load_and_process_audio_from_array(audio, sample_rate, max_length=16000):
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav

    wav = scipy_resample(audio, sample_rate)
    wav = wav[:max_length] if len(wav) > max_length else np.pad(wav, (0, max_length - len(wav)))
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    wav = tf.reshape(wav, (max_length, 1))
    return wav

#function to plot waveform of audio
def plot_wave_form(wave_form):
    plt.figure(figsize=(10, 6))
    plt.plot(wave_form)
    plt.title('Waveform')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()

#function to convert waveform into spectrogram using STFT
def spectrogram(wave_form):
    wave_form = tf.reshape(wave_form, [-1, 16000])
    spect = tf.signal.stft(wave_form, frame_length=256, frame_step=128)
    spect = tf.abs(spect)
    spect = spect[..., tf.newaxis]
    return spect

def plot_spectrogram(spectrogram):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram + np.finfo(float).eps)
    plt.figure(figsize=(10, 6))
    plt.imshow(log_spec, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (Frames)")
    plt.ylabel("Frequency (Bins)")
    plt.title("Spectrogram")
    plt.show()

def get_labels(dataset_dir):
    extracted_path = os.path.join(dataset_dir, 'extracted')
    data_dir = pathlib.Path(extracted_path)
    all_labels = [path.parent.name for path in data_dir.glob('*/**/*.wav')]
    unique_labels = list(set(all_labels))
    return unique_labels

#function to record audio from the microphone
def record_audio(duration=2, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  #wait until recording is finished
    print("Recording complete.")
    audio = audio.flatten()  #flatten to 1D array
    return audio, sample_rate

#load pretrained model and custom channel attention layer
model = tf.keras.models.load_model(model_path, custom_objects={'ChannelAttention': ChannelAttention})

#get unique labels from dataset and encode them
all_labels = get_labels(dataset_dir)
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

#record audio and preprocess
recorded_audio, sr = record_audio(duration=2)  #record 2 seconds of audio
processed_audio = load_and_process_audio_from_array(recorded_audio, sr)

plot_wave_form(processed_audio.numpy())

#convert to spectrogram
test_spectrogram = spectrogram(processed_audio)
print(test_spectrogram.shape)

#remove batch dimension for plotting
test_spectrogram_plot = np.squeeze(test_spectrogram, axis=0)
plot_spectrogram(test_spectrogram_plot)

#make prediction
prediction = model.predict(test_spectrogram)

#extract the predicted label and its confidence
predicted_label_index = np.argmax(prediction, axis=-1)[0]
confidence = prediction[0][predicted_label_index]  #probability of the predicted label

predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

#print predicted label and confidence
print(f"Predicted Label: {predicted_label}, Confidence: {confidence:.2f}")