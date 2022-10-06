from AudioDataset import AudioDataset

import torchaudio

import torch


def transformData(audio_paths, transformParams):

    transformedDataset = AudioDataset(audio_paths)
    print(transformParams)
    for transformAudio, transformSpec in transformParams:

        audio_train_dataset = AudioDataset(audio_paths,
                                           specTransformList=transformSpec,
                                           audioTransformList=transformAudio)

        transformedDataset = torch.utils.data.ConcatDataset(
            [transformedDataset, audio_train_dataset])

    return transformedDataset