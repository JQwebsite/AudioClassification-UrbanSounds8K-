from pathlib import Path
import Augmentation
from AudioDataset import AudioDataset
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

config = ConfigParser()
config.read('config.ini')

audio_paths = Augmentation.getAudioPaths('./data/')[0:10]

test_len = int(int(config['data']['train_percent']) / 100 * len(audio_paths))
audio_train_paths, audio_val_paths = audio_paths[:test_len], audio_paths[
    test_len:]

transformList = [
    [
        torchaudio.transforms.TimeMasking(80),
    ],
    [
        torchaudio.transforms.FrequencyMasking(80),
    ],
    [
        torchaudio.transforms.TimeMasking(80),
        torchaudio.transforms.FrequencyMasking(80),
    ],
]

audio_train_dataset = transforms.transformData(audio_train_paths,
                                               transformList)

audio_val_dataset = AudioDataset(audio_val_paths)

train_dataloader = torch.utils.data.DataLoader(
    audio_train_dataset,
    batch_size=4,
    num_workers=0,
    shuffle=True,
)

val_dataloader = torch.utils.data.DataLoader(
    audio_val_dataset,
    batch_size=4,
    num_workers=0,
    shuffle=True,
)

model = ResNet18

mymodel = machineLearning(model, train_dataloader, val_dataloader,
                          config['logger'])

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=float(config['model']['learning_rate']))

epochs = int(config['model']['num_epochs'])
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}\n-------------------------------')
    train_loss, train_accuracy = mymodel.train(cost, optimizer)
    val_loss, val_accuracy = mymodel.val(cost)
    print(f'Training | Loss: {train_loss} Accuracy: {train_accuracy}%')
    print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')
    mymodel.tensorBoardLogging(train_loss, train_accuracy, val_loss,
                               val_accuracy, epoch)

mymodel.saveModel()