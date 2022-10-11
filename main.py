from pathlib import Path
import Augmentation
from AudioDataset import transformData
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
import machineLearning
from model import ResNet18, CNNNetwork, M5
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    audio_paths = Augmentation.getAudioPaths('./data/')

    test_len = int(
        int(config['data']['train_percent']) / 100 * len(audio_paths))
    audio_train_paths, audio_val_paths = audio_paths[:test_len], audio_paths[
        test_len:]

    transformList = [
        {
            "audio": [
                AddGaussianNoise(min_amplitude=0.001,
                                 max_amplitude=0.015,
                                 p=0.5),
                TimeStretch(min_rate=0.8,
                            max_rate=1.2,
                            p=0.5,
                            leave_length_unchanged=False),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ],
            "spectrogram": []
        },
        {
            "audio": [],
            "spectrogram": [
                torchaudio.transforms.TimeMasking(80),
                torchaudio.transforms.FrequencyMasking(80)
            ],
        },
    ]

    audio_train_dataset = transformData(audio_train_paths, transformList)

    audio_val_dataset = transformData(audio_val_paths)

    print(
        f'Train dataset Length: {len(audio_train_dataset)} ({len(audio_train_paths)} before augmentation)'
    )
    print(f'Validation dataset Length: {len(audio_val_dataset)}')

    train_dataloader = torch.utils.data.DataLoader(
        audio_train_dataset,
        batch_size=int(config['model']['batch_size']),
        num_workers=int(config['model']['num_workers']),
        shuffle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        audio_val_dataset,
        batch_size=int(config['model']['batch_size']),
        num_workers=int(config['model']['num_workers']),
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18.to(device)

    # set model parameters
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config['model']['learning_rate']))
    epochs = int(config['model']['num_epochs'])

    title = config['logger']['title'] if config['logger'][
        'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")

    if config['logger'].getboolean('master_logger'):
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
