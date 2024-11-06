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

#set up directories
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/audio_processing')
dataset_dir = os.path.join(base_dir, 'dataset')
dataset_path = os.path.join(dataset_dir, 'dataset_commands-002.gz')
models_dir = os.path.join(base_dir, 'model')
logs_dir = os.path.join(base_dir, 'logs')

def read_file(gz_path):
    extracted_path = os.path.join(dataset_dir, 'extracted')

    #extract and unpack dataset
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path + '.tar', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    shutil.unpack_archive(extracted_path + '.tar', extracted_path)
    
    data_dir = pathlib.Path(extracted_path)

    #collect audio paths and labels
    all_audio_paths = [str(path) for path in data_dir.glob('*/**/*.wav')]
    all_labels = [path.parent.name for path in data_dir.glob('*/**/*.wav')]

    return all_audio_paths, all_labels

def load_and_process_audio(filename, max_length=16000):
    #load and decode the audio file
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    
    #squeeze out the redundant channel dimension
    wav = tf.squeeze(wav, axis=-1)
    
    #function to resample to 16000 hz if needed
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav
    
    #apply resampling using py_function to keep tensorflow graph compatibility
    wav = tf.py_function(scipy_resample, [wav, sample_rate], tf.float32)

    #pad or truncate to ensure max_length of 16000 samples
    wav = wav[:max_length] if tf.shape(wav)[0] > max_length else tf.pad(wav, [[0, max_length - tf.shape(wav)[0]]])

    #reshape to (16000, 1) to match conv1d input expectations
    wav = tf.reshape(wav, (max_length, 1))
    
    return wav

def process_path(file_path, label):
    #load and process audio to get (16000, 1) shape
    audio = load_and_process_audio(file_path)
    return audio, label

def paths_and_labels_to_dataset(audio_paths, labels):
    #create dataset of audio file paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    
    #zip the two datasets together and map with process_path
    audio_label_ds = tf.data.Dataset.zip((path_ds, label_ds))
    return audio_label_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

def prepare_for_training(ds, batch_size=32, shuffle_buffer_size=1000):
    #shuffle, batch, and prefetch the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def plot_history(history):
    #plot training and validation accuracy and loss
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

#read and encode audio paths and labels
all_audio_paths, all_labels = read_file(dataset_path)

#encode labels
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

#split into training and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_audio_paths, all_labels_encoded, test_size=0.02, random_state=42, stratify=all_labels_encoded
)

#create and prepare datasets
train_dataset = paths_and_labels_to_dataset(train_paths, train_labels)
val_dataset = paths_and_labels_to_dataset(val_paths, val_labels)

train_dataset = prepare_for_training(train_dataset)
val_dataset = prepare_for_training(val_dataset)

#define the model
model_time_domain = models.Sequential([
    layers.Input(shape=(16000, 1)),
    layers.Conv1D(16, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(36, activation='softmax')
])

#compile the model
model_time_domain.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#set up callbacks for saving the model, early stopping, and tensorboard logging
model_save_path = os.path.join(models_dir, "speech_to_text.keras")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_loss')
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)

#train the model
history_time_domain = model_time_domain.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
)

#plot training history
plot_history(history_time_domain)
