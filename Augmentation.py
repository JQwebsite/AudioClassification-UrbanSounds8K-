import torch
import torchaudio
import random


class Augment():

    def __init__(self):
        pass

    def pad_trunc(self, aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = int(sr / 1000 * max_ms)

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

    def rechannel(self, aud, new_channel):
        sig, sr = aud
        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud
        elif (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])
        return ((resig, sr))

    def resample(self, aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            # Nothing to do
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])
        return ((resig, newsr))