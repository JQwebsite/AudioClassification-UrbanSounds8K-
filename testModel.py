from pathlib import Path
import Augmentation
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import utils
import os
import transforms
from machineLearning import machineLearning
from model import ResNet18
from configparser import ConfigParser
import matplotlib.pyplot as plt


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = ResNet18
    # state_dict = torch.load("saved_model/soundclassifier.pth")
    # cnn.load_state_dict(state_dict)
    # load urban sound dataset
    mel_spectrogram = torchaudio.transforms.Spectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, "cpu")
    # get a sample from the us dataset for inference
    input, target = usd[0][0], usd[0][1]  # [num_cha, fr, t]
    input.unsqueeze_(0)
    # make an inference
    predicted, expected = predict(cnn, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")