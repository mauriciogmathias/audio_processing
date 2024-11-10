import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import os
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#set tensorflow threading options for cpu optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#set up paths for dataset, model, and logs
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/audio_processing')
dataset_dir = os.path.join(base_dir, 'dataset')
dataset_path = os.path.join(dataset_dir, 'dataset_commands-002.gz')
models_dir = os.path.join(base_dir, 'model')
logs_dir = os.path.join(base_dir, 'logs')

#custom layer to apply channel attention mechanism
class ChannelAttention(tf.keras.layers.Layer):
    #initialize layer with specified ratio for dimensionality reduction
    def __init__(self, ratio = 8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D() #global avg pooling
        self.max_pool = layers.GlobalMaxPooling2D()     #global max pooling

    #define the layer structure based on input shape
    def build(self, input_shape):
        #fully connected layers: first layer reduces dimension using ratio
        self.fc1 = layers.Dense(units = input_shape[-1] // self.ratio, activation = 'relu')
        #second layer restores original dimension with sigmoid activation for scaling
        self.fc2 = layers.Dense(units = input_shape[-1], activation = 'sigmoid')

    #apply attention mechanism to input
    def call(self, inputs):
        #average and max pooling to capture global spatial information
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        
        #pass pooled outputs through fully connected layers
        avg_out = self.fc2(self.fc1(avg_out))
        max_out = self.fc2(self.fc1(max_out))
        
        #combine outputs and reshape to match input dimensions
        out = avg_out + max_out
        out = tf.expand_dims(tf.expand_dims(out, axis = 1), axis = 1)
        
        #scale input features by channel attention output
        return inputs * out

#function to read, unpack, and prepare audio dataset paths and labels
def read_file(gz_path):
    extracted_path = os.path.join(dataset_dir, 'extracted')
    
    #extract and unpack dataset from .gz and .tar formats
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path + '.tar', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    shutil.unpack_archive(extracted_path + '.tar', extracted_path)
    
    data_dir = pathlib.Path(extracted_path)
    
    #collect audio paths and labels based on file structure
    all_audio_paths = [str(path) for path in data_dir.glob('*/**/*.wav')]
    all_labels = [path.parent.name for path in data_dir.glob('*/**/*.wav')]
    
    return all_audio_paths, all_labels

#function to load and preprocess audio to standard length, sampling, and shape
def load_and_process_audio(filename, max_length=16000):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    
    #resample audio to 16000Hz
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav
    wav = tf.py_function(scipy_resample, [wav, sample_rate], tf.float32)
    
    wav = wav[:max_length] if tf.shape(wav)[0] > max_length else tf.pad(wav, [[0, max_length - tf.shape(wav)[0]]])
    
    wav = tf.reshape(wav, (max_length, 1))
    
    return wav

def process_path(file_path, label):
    audio = load_and_process_audio(file_path)
    return audio, label

def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    audio_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return audio_label_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

#function to convert audio waveform into spectrograms using Short-Time Fourier Transform (STFT)
def spectrogram(wave_form):
    #flatten any extra dimensions and ensure waveform length of 16000
    wave_form = tf.reshape(wave_form, [-1, 16000])

    #STFT applied to decompose the audio waveform into its frequency components over time.
    #STFT divides the audio signal into short overlapping segments using a frame_length of 256 samples
    #frame_length determines the segment size and impacts the frequency resolution
    #each frame is shifted by frame_step (128 samples here), controlling how much each frame overlaps with the next
    #these parameters together determine the time-frequency resolution balance of the resulting spectrogram
    spect = tf.signal.stft(wave_form, frame_length=256, frame_step=128)
    spect = tf.abs(spect) #convert complex STFT output to magnitude spectrum for analysis

    #add channel dimension to the spectrogram for Conv2D compatibility
    spect = spect[..., tf.newaxis]
    
    return spect

def get_spectrogram_and_label_id(audio, label):
    spect = spectrogram(audio)
    return spect, label

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

all_audio_paths, all_labels = read_file(dataset_path)
num_labels = len(np.unique(all_labels))

label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_audio_paths, all_labels_encoded, test_size=0.02, random_state=42, stratify=all_labels_encoded
)

train_dataset = paths_and_labels_to_dataset(train_paths, train_labels)
val_dataset = paths_and_labels_to_dataset(val_paths, val_labels)
train_dataset = prepare_for_training(train_dataset)
val_dataset = prepare_for_training(val_dataset)

train_spec = train_dataset.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)
val_spec = val_dataset.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=tf.data.AUTOTUNE)

#define normalization layer and adapt it to the spectrogram data
norm_layer = tf.keras.layers.Normalization()
for aux_spect, _ in train_spec.take(1):
    norm_layer.adapt(aux_spect)
    input_shape = aux_spect.shape[1:]

#model definition
model_spectrogram = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),                        #resize input to fixed dimensions
    norm_layer,                                     #apply normalization to input data
    layers.Conv2D(32, 3, activation='relu'),        #convolution layer with 32 filters
    ChannelAttention(ratio = 8),                    #apply channel attention after Conv2D layer
    layers.Conv2D(64, 3, activation='relu'),        #second convolution layer with 64 filters
    ChannelAttention(ratio = 8),                    #apply another channel attention
    layers.MaxPooling2D(),                          #reduce feature map dimensions
    layers.Dropout(0.25),                           #apply dropout for regularization
    layers.Flatten(),                               #flatten feature maps for dense layer
    layers.Dense(128, activation='relu'),           #fully connected dense layer
    layers.Dropout(0.5),                            #dropout for dense layer regularization
    layers.Dense(num_labels, activation='softmax')  #output layer with softmax activation
])

model_spectrogram.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#set up model saving path and training callbacks for checkpointing and early stopping
model_save_path = os.path.join(models_dir, "speech_to_text.keras")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only = True, monitor = 'val_loss')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = 8, restore_best_weights = True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = logs_dir)

history_freq_time_domain = model_spectrogram.fit(
    train_spec,
    epochs = 20,
    validation_data = val_spec,
    callbacks = [checkpoint_cb, early_stopping_cb, tensorboard_cb]
)

plot_history(history_freq_time_domain)
