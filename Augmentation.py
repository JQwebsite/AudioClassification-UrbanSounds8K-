import torch
import torchaudio
import random
from torch.utils.data import Dataset, DataLoader
from configparser import ConfigParser
from pathlib import Path
import os


class Augmentor():

    def __init__(self):
        config = ConfigParser()
        config.read('config.ini')
        self.audio_duration = int(config['augmentations']['duration'])
        self.audio_channels = int(config['augmentations']['num_channels'])
        self.audio_sampling = int(config['augmentations']['sample_rate'])

    def audio_preprocessing(self, audioIn):
        return self.resample(self.rechannel(audioIn))

    def pad_trunc(self, aud):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = int(sr / 1000 * self.audio_duration)

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return (sig, sr)

    def rechannel(self, aud):
        sig, sr = aud
        if (sig.shape[0] == self.audio_channels):
            # Nothing to do
            return aud
        elif (self.audio_channels == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])
        return ((resig, sr))

    def resample(self, aud):
        sig, sr = aud
        if (sr == self.audio_sampling):
            # Nothing to do
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr,
                                               self.audio_sampling)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, self.audio_sampling)(
                sig[1:, :])
            resig = torch.cat([resig, retwo])
        return ((resig, self.audio_sampling))


def getAudioPaths(main_path):
    audio_paths = []
    for path in [str(p) for p in Path(main_path).glob('fold*')]:
        for wav_path in [str(p) for p in Path(path).glob(f'*.wav')]:
            audio_paths.append(wav_path)
    return audio_paths


def getAudio(path):
    wav_paths = []
    for wav_path in [str(p) for p in Path(path).glob(f'*.wav')]:
        wav_paths.append(wav_path)
    return wav_paths