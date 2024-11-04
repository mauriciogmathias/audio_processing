import tensorflow as tf
import pathlib
import os
import gzip
import shutil
import numpy as np
import matplotlib.pyplot as plt

#set tensorflow threading options for cpu optimization
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

#set up directories
base_dir = os.path.join(os.path.expanduser('~'), 'Desktop/ml/audio_processing')
dataset_dir = os.path.join(base_dir, 'dataset')
dataset_path = os.path.join(dataset_dir, 'dataset_commands-002.gz')

def read_file(gz_path):
    extracted_path = os.path.join(dataset_dir, 'extracted')

    with gzip.open(gz_path, 'rb') as f_in:
        with open(extracted_path + '.tar', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    #extract resulting file
    shutil.unpack_archive(extracted_path + '.tar', extracted_path)
    
    data_dir = pathlib.Path(extracted_path)

    all_audio_paths = list(data_dir.glob('*/**/*.wav'))
    all_labels = [path.parent.name for path in all_audio_paths]

    return all_audio_paths, all_labels

all_audio_paths, all_labels = read_file(dataset_path)

np.unique(all_labels)

np.unique(all_labels).shape

example_audio_path = all_audio_paths[0]

audio_binary = tf.io.read_file(str(example_audio_path))
audio, _ = tf.audio.decode_wav(audio_binary)
audio = tf.squeeze(audio, axis = -1)

plt.figure(figsize=(10, 6))
plt.plot(audio.numpy())
plt.title(f'Forma de onda para {example_audio_path}')
plt.xlabel('Amostras')
plt.ylabel('Amplitude')
plt.show()
