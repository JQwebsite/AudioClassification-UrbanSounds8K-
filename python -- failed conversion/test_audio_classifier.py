import AudioAugmentation
from pathlib import Path
import torchaudio


def test_load():
    walker = sorted(str(p) for p in Path("./test/").glob(f'*.wav'))
    for i, file_path in enumerate(walker):
        assert torchaudio.load(file_path)


def test_pad_trunc():
    target_length = 4
    assert len(
        AudioAugmentation.pad_trunc(torchaudio.load('./test/1_44100_0830.wav'),
                                    target_length *
                                    1000)[0][0]) == 44100 * target_length
    assert len(
        AudioAugmentation.pad_trunc(torchaudio.load('./test/2_44100_2250.wav'),
                                    target_length *
                                    1000)[0][0]) == 44100 * target_length
    assert len(
        AudioAugmentation.pad_trunc(torchaudio.load('./test/2_44100_4000.wav'),
                                    target_length *
                                    1000)[0][0]) == 44100 * target_length


def test_rechannel():
    target_channel = 1
    assert AudioAugmentation.rechannel(
        torchaudio.load('./test/1_44100_0830.wav'),
        target_channel)[0].shape[0] == target_channel
    assert AudioAugmentation.rechannel(
        torchaudio.load('./test/2_44100_4000.wav'),
        target_channel)[0].shape[0] == target_channel


def test_resample():
    target_sr = 44100
    assert len(
        AudioAugmentation.resample(torchaudio.load('./test/1_11025_4000.wav'),
                                   target_sr)[0][0]) == 4 * target_sr
    assert len(
        AudioAugmentation.resample(torchaudio.load('./test/1_44100_0830.wav'),
                                   target_sr)[0][0]) == 0.830 * target_sr
    assert len(
        AudioAugmentation.resample(torchaudio.load('./test/1_96000_0310.wav'),
                                   target_sr)[0][0]) == 0.310 * target_sr
