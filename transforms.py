from AudioDataset import AudioDataset
import torchaudio
import torch


def transformData(audio_paths, transformParams):
    # outputs original waveform + transformParams
    transformedDataset = AudioDataset(audio_paths)

    for transform in transformParams:
        audio_train_dataset = AudioDataset(
            audio_paths,
            specTransformList=transform['spectrogram'],
            audioTransformList=transform['audio'])

        transformedDataset = torch.utils.data.ConcatDataset(
            [transformedDataset, audio_train_dataset])

    return transformedDataset