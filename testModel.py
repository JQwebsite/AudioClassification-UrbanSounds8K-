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
import machineLearning
from model import ResNet18
from configparser import ConfigParser
import matplotlib.pyplot as plt
from AudioDataset import AudioDataset
import Augmentation


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
    state_dict = torch.load("saved_model/cnn.pth")
    cnn.load_state_dict(state_dict)
    # load urban sound dataset
    audio_paths = Augmentation.getAudioPaths_test('./data/')
    # get a sample from the us dataset for inference
    audio_val_dataset = AudioDataset(audio_paths)
    X, y = audio_val_dataset[0]
    