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
import machineLearning
from model import ResNet18, CNNNetwork, M5
from configparser import ConfigParser
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    audio_paths = Augmentation.getAudioPaths('./data/')

    test_len = int(
        int(config['data']['train_percent']) / 100 * len(audio_paths))
    audio_train_paths, audio_val_paths = audio_paths[:test_len], audio_paths[
        test_len:]

    transformList = [
        [[torchaudio.transforms.Vol(1.1)], []],
        [[torchaudio.transforms.Vol(0.9)], []],
        [[], [torchaudio.transforms.TimeMasking(50)]],
        [[], [torchaudio.transforms.FrequencyMasking(50)]],
        [[],
         [
             torchaudio.transforms.TimeMasking(80),
             torchaudio.transforms.FrequencyMasking(80)
         ]],
    ]

    audio_train_dataset = transforms.transformData(audio_train_paths,
                                                   transformList)

    audio_val_dataset = AudioDataset(audio_val_paths)

    train_dataloader = torch.utils.data.DataLoader(
        audio_train_dataset,
        batch_size=int(config['model']['batch_size']),
        num_workers=4,
        shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        audio_val_dataset,
        batch_size=int(config['model']['batch_size']),
        num_workers=4,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18.to(device)

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config['model']['learning_rate']))

    epochs = int(config['model']['num_epochs'])
    title = config['logger']['title'] if config['logger'][
        'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")
    if int(config['logger']['master_logger']):

        writer = SummaryWriter(f'./logs/{title}')
        writer.close()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}\n-------------------------------')
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, cost, optimizer, device)
        val_loss, val_accuracy = machineLearning.val(model, val_dataloader,
                                                     cost, device)
        print(f'Training | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')
        if int(config['logger']['log_iter_params']) and int(
                config['logger']['master_logger']):
            machineLearning.tensorBoardLogging(writer, train_loss,
                                               train_accuracy, val_loss,
                                               val_accuracy, epoch,
                                               config['logger'])

    torch.save(model.state_dict, f'saved_model/{title}.pt')
