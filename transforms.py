from AudioDataset import AudioDataset
import torchaudio
import torch


def transformData(audio_paths, transformParams):

    transformedDataset = AudioDataset(audio_paths)

    # if int(config['data']['augment_data']):
    for transforms in transformParams:
        audio_train_dataset = AudioDataset(audio_paths,
                                           transformList=transforms)
        transformedDataset = torch.utils.data.ConcatDataset(
            [transformedDataset, audio_train_dataset])

    return transformedDataset