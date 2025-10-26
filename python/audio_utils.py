"""
Module for audio utilities in live whisper
"""

import time
import threading

import numpy as np
import soundfile as sf
import scipy
import torch
import torch.nn.functional as F

def ensure_sample_rate(
    waveform: np.array, original_sample_rate: int, desired_sample_rate: int = 16000
) -> tuple[np.array, float]:
    """Ensure the waveform's sample rate, resample if necessary"""
    if original_sample_rate != desired_sample_rate:
        print(f"resample_audio: {original_sample_rate} HZ -> {desired_sample_rate} HZ")
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate


def ensure_channels(
    waveform: np.array, original_channels: int, desired_channels=1
) -> tuple[np.array, int]:
    """Ensure the waveform is single channel, take mean of channels if necessary"""
    assert (
        desired_channels == 1
    )  # this function only works with 1 desired channel TODO rename or fix
    if original_channels != desired_channels:
        print(f"convert_channels: {original_channels} -> {desired_channels}")
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels


def pad_or_trim(audio_array: np.ndarray, n_mels: int, max_length: int):
    """
    Put an audio mels spectogram of arbitrary length
    """
    x_mel = np.zeros((n_mels, max_length), dtype=np.float32)
    real_length = (
        audio_array.shape[1] if audio_array.shape[1] <= max_length else max_length
    )
    x_mel[:, :real_length] = audio_array[:, :real_length]

    return x_mel


def log_mel_spectrogram(
    audio: torch.Tensor | np.ndarray,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    padding: int = 0,
) -> torch.Tensor:
    """
    Get log mel spectrogram of a waveform, using filters in mel_80_filters.txt
    """
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(n_fft)

    stft = torch.stft(audio, n_fft, hop_length, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def mel_filters(n_mels: int) -> torch.Tensor:
    """
    Load mel filters tensor from model/mel_80_filters.txt
    """
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    filters_path = "../model/mel_80_filters.txt"
    mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return torch.from_numpy(mels_data)


class FakeInputStream:
    """Class for a audio stream like sounddevice.InputStream, using a .wav file as input"""
    def __init__(self, file_path, callback, blocksize=1024):
        self.file_path = file_path
        self.callback = callback
        self.blocksize = blocksize
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.join()

    def _run(self):
        # TODO use time.time to get time based reading
        audio_data, sample_rate = sf.read(self.file_path, dtype="float32")
        channels = audio_data.ndim
        audio_data, channels = ensure_channels(audio_data, channels)
        audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
        for i in range(0, len(audio_data), self.blocksize):
            time.sleep(self.blocksize / sample_rate)
            block = audio_data[i : i + self.blocksize]
            self.callback(block.reshape(-1, 1), len(block), None, None)
