import Augmentation
import torchaudio
import torch


def transformData(audio_paths, transformParams):

    transformedDataset = Augmentation.AudioDataset(audio_paths)

    # if int(config['data']['augment_data']):
    for transforms in transformParams:
        audio_train_dataset = Augmentation.AudioDataset(
            audio_paths, transformList=transforms)
        transformedDataset = torch.utils.data.ConcatDataset(
            [transformedDataset, audio_train_dataset])

    return transformedDataset