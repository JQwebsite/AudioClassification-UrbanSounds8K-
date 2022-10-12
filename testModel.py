from pathlib import Path
import Augmentation
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = ResNet18.to(device)    
    state_dict = torch.load("saved_model/cnn.pth", map_location=device)
    cnn.load_state_dict(state_dict)
    # load urban sound dataset
    audio_paths = Augmentation.getAudio('./data/testset')
    # get a sample from the us dataset for inference
    audio_test_dataset = AudioDataset(audio_paths)
    test_dataloader = torch.utils.data.DataLoader(
        audio_test_dataset,
        batch_size=12,
        num_workers=0,
        shuffle=True,
    )

    test_loss, test_acc = (machineLearning.val(cnn, test_dataloader, torch.nn.CrossEntropyLoss(), device))
    print(f'Validating  | Loss: {test_loss} Accuracy: {test_acc}% \n')