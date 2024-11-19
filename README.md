# audio_processing

This project is my experimentation with creating a sequential model to recognize spoken commands. It contains scripts for creating/training the model, testing, and making predictions on spoken command recognition. The model is designed to classify audio commands from spectrogram representations of the audio data.

The dataset can be found at [this link](https://cdn3.gnarususercontent.com.br/3981-tensorflow-keras/Projeto/dataset_commands-002.gz).

---

## Project Structure

- **`training.py`**: Script to create a sequential model and train it on the speech command dataset.
- **`test.py`**: Script to test the trained model using a predefined audio file.
- **`record_and_predict.py`**: Script to record audio from a microphone and make predictions using the trained model.

