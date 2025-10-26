
import torch
import scipy
import numpy as np

import matplotlib.pyplot as plt

def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        print(
            "resample_audio: {} HZ -> {} HZ".format(
                original_sample_rate, desired_sample_rate
            )
        )
        desired_length = int(
            round(float(len(waveform)) / original_sample_rate * desired_sample_rate)
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate


def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels != desired_channels:
        print("convert_channels: {} -> {}".format(original_channels, desired_channels))
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels


def get_char_index(c):
    if "A" <= c <= "Z":
        return ord(c) - ord("A")
    elif "a" <= c <= "z":
        return ord(c) - ord("a") + (ord("Z") - ord("A") + 1)
    elif "0" <= c <= "9":
        return ord(c) - ord("0") + (ord("Z") - ord("A")) + (ord("z") - ord("a")) + 2
    elif c == "+":
        return 62
    elif c == "/":
        return 63
    else:
        print(f"Unknown character {ord(c)}, {c}")
        exit(-1)


def base64_decode(encoded_string):
    if not encoded_string:
        print("Empty string!")
        exit(-1)

    output_length = len(encoded_string) // 4 * 3
    decoded_string = bytearray(output_length)

    index = 0
    output_index = 0
    while index < len(encoded_string):
        if encoded_string[index] == "=":
            return " "

        first_byte = (get_char_index(encoded_string[index]) << 2) + (
            (get_char_index(encoded_string[index + 1]) & 0x30) >> 4
        )
        decoded_string[output_index] = first_byte

        if index + 2 < len(encoded_string) and encoded_string[index + 2] != "=":
            second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0F) << 4) + (
                (get_char_index(encoded_string[index + 2]) & 0x3C) >> 2
            )
            decoded_string[output_index + 1] = second_byte

            if index + 3 < len(encoded_string) and encoded_string[index + 3] != "=":
                third_byte = (
                    (get_char_index(encoded_string[index + 2]) & 0x03) << 6
                ) + get_char_index(encoded_string[index + 3])
                decoded_string[output_index + 2] = third_byte
                output_index += 3
            else:
                output_index += 2
        else:
            output_index += 1

        index += 4

    return decoded_string.decode("utf-8", errors="replace")


def read_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        vocab = {}
        for line in f:
            if len(line.strip().split(" ")) < 2:
                key = line.strip().split(" ")[0]
                value = ""
            else:
                key, value = line.strip().split(" ")
            vocab[key] = value
    return vocab


def mel_filters(n_mels):
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    filters_path = "../model/mel_80_filters.txt"
    mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return torch.from_numpy(mels_data)

