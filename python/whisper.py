import time
from matplotlib import pyplot as plt
import numpy as np

# from rknn.api import RKNN
# comment this when using RKNN:
class RKNN:
    """placeholder to shup up the linter when not using rknn"""
    def load_rknn(self, model):
        return 0
    def init_runtime(self, target, device_id):
        return 0

import argparse
import soundfile as sf
import onnxruntime
import torch
import torch.nn.functional as F
import sounddevice as sd

from file_audio_stream import FakeInputStream
from util import base64_decode, mel_filters, read_vocab
from merge import merge_segments


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_LENGTH = CHUNK_LENGTH * 100  # CHUNK_LENGTH * SAMPLE_RATE / HOP_LENGTH
N_MELS = 80
# wait 5 seconds between each transcription
TRANSCRIBE_INTERVAL = 5

audio_buffer = np.zeros(0, dtype=np.float32)
buffer_offset = 0



def pad_or_trim(audio_array):
    """
    Put an audio mels spectogram of arbitrary length
    """
    x_mel = np.zeros((N_MELS, MAX_LENGTH), dtype=np.float32)
    real_length = (
        audio_array.shape[1] if audio_array.shape[1] <= MAX_LENGTH else MAX_LENGTH
    )
    x_mel[:, :real_length] = audio_array[:, :real_length]

    return x_mel


def log_mel_spectrogram(audio, n_mels, padding=0):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT)

    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def run_encoder(encoder_model, in_encoder):
    out_encoder = None
    if "rknn" in str(type(encoder_model)):
        out_encoder = encoder_model.inference(inputs=in_encoder)[0]
    elif "onnx" in str(type(encoder_model)):
        out_encoder = encoder_model.run(None, {"x": in_encoder})[0]

    return out_encoder


def _decode(decoder_model, tokens, out_encoder):
    out_decoder = None
    if "rknn" in str(type(decoder_model)):
        out_decoder = decoder_model.inference(
            [np.asarray([tokens], dtype="int64"), out_encoder]
        )[0]
    elif "onnx" in str(type(decoder_model)):
        out_decoder = decoder_model.run(
            None, {"tokens": np.asarray([tokens], dtype="int64"), "audio": out_encoder}
        )[0]

    return out_decoder


def run_decoder(decoder_model, out_encoder, vocab, task_code):
    # tokenizer = whisper.decoding.get_tokenizer( True, #model.is_multilingual
    #                                             task="transcribe",
    #                                             language="en",
    #                                             )

    end_token = 50257  # tokenizer.eot
    next_token = 50258  # tokenizer.sot
    transcribe_token = 50359
    notimestamps_token = 50363
    # no notimestamps token: lets see
    timestamp_begin = 50364  # tokenizer.timestamp_begin
    tokens = [
        next_token,
        task_code,
        transcribe_token,
        notimestamps_token,
    ]  # use timestamp_begin at the end to get timestamps
    preamble_len = len(tokens)

    max_tokens = 48
    tokens_str = ""
    pop_id = max_tokens

    tokens = tokens * int(max_tokens / preamble_len)

    start = time.time()
    while next_token != end_token:
        out_decoder = _decode(decoder_model, tokens, out_encoder)
        next_token = out_decoder[0, -1].argmax()
        next_token_str = vocab[str(next_token)]
        # print(f"{next_token_str}: {next_token}")
        tokens.append(next_token)

        if next_token == end_token:
            tokens.pop(-1)
            next_token = tokens[-1]
            break
        if next_token > timestamp_begin:
            pass
            # print("timestamp detected:", next_token)
            # continue
        if pop_id > preamble_len:
            pop_id -= 1
        else:
            print("=== Buffer full !")
        if time.time() - start > CHUNK_LENGTH/2:
            print("=== Decoder fucked! :")
            return ""

        tokens.pop(pop_id)
        tokens_str += next_token_str

    result = (
        tokens_str.replace("\u0120", " ").replace("<|endoftext|>", "").replace("\n", "")
    )
    if task_code == 50260:  # TASK_FOR_ZH
        result = base64_decode(result)
    return result


def init_model(model_path, target=None, device_id=None):
    if model_path.endswith(".rknn"):
        # Create RKNN object
        model = RKNN()

        # Load RKNN model
        print("--> Loading model")
        ret = model.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model "{}" failed!'.format(model_path))
            exit(ret)
        print("done")

        # init runtime environment
        print("--> Init runtime environment")
        ret = model.init_runtime(target=target, device_id=device_id)
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("done")

    elif model_path.endswith(".onnx"):
        model = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    return model


def release_model(model):
    if "rknn" in str(type(model)):
        model.release()
    elif "onnx" in str(type(model)):
        del model
    model = None


def load_array_from_file(filename):
    with open(filename, "r") as file:
        data = file.readlines()

    array = []
    for line in data:
        row = [float(num) for num in line.split()]
        array.extend(row)

    return np.array(array).reshape((80, 2000))

audio_offset = 0.0
audio_len = 0.0
def audio_callback(indata, frames, time_info, status):
    global audio_buffer, buffer_offset, audio_offset, audio_len
    if status:
        print("Status:", status)

    # take first channel
    indata = indata[:, 0]
    # append new chunk
    audio_buffer = np.append(audio_buffer, indata)
    audio_len = len(audio_buffer) / SAMPLE_RATE
    # clip
    clip_start = max(0, len(audio_buffer) - N_SAMPLES)
    audio_offset += clip_start / SAMPLE_RATE
    buffer_offset += clip_start
    audio_buffer = audio_buffer[clip_start:-1]

# TODO put this somewhere better
results_buffer = []

debug_counter = 0
def process_audio(data: np.array, encoder_model, decoder_model, vocab, task_code):
    global debug_counter, audio_buffer, buffer_offset, audio_offset, audio_len # TODO remove globals
    # Try to transcribe from buffer
    audio_array = log_mel_spectrogram(audio_buffer, N_MELS).numpy()
    x_mel = pad_or_trim(audio_array)

    # plt.figure(figsize=(10, 4))
    # plt.imshow(x_mel, origin="lower", aspect="auto", interpolation="nearest")
    # plt.title("Mel Spectrogram")
    # plt.xlabel("Time Frames")
    # plt.ylabel("Mel Bands")
    # plt.colorbar(label="Log Power")
    # plt.tight_layout()
    # plt.show()

    debug_counter += 1
    sf.write(f'logs/log_{debug_counter}.wav', audio_buffer, SAMPLE_RATE)
    print(f"==== Frame: {debug_counter}")

    x_mel = np.expand_dims(x_mel, 0)

    out_encoder = run_encoder(encoder_model, x_mel)
    result = run_decoder(decoder_model, out_encoder, vocab, task_code)
    results_buffer.append((audio_offset, audio_offset + audio_len, result))
    merged = merge_segments(results_buffer)
    print("Whisper output:", result)
    print("Merged:", merged)

def transcribe_live(
    vocab: dict,
    task_code: int,
    input_file: str | None,
    encoder_model,
    decoder_model,
):
    """
    Live transcription loop
    """
    global audio_buffer

    if input_file is None:
        stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=1024,
            callback=audio_callback,
        )
    else:
        stream = FakeInputStream(input_file, callback=audio_callback)

    with stream:
        print("Recording... press Ctrl+C to stop.")
        last_time = time.time()

        while True:
            now = time.time()
            if now - last_time >= TRANSCRIBE_INTERVAL:
                last_time = now
                # make a copy of the buffer for thread safety
                data_copy = np.copy(audio_buffer)
                process_audio(data_copy, encoder_model, decoder_model, vocab, task_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Python Demo", add_help=True)
    # basic params
    parser.add_argument(
        "--encoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--input", type=str, required=False, help="target RKNPU platform"
    )
    parser.add_argument(
        "--task", type=str, required=True, help="recognition task, could be en or zh"
    )
    parser.add_argument(
        "--target", type=str, default="rk3588", help="target RKNPU platform"
    )
    parser.add_argument("--device_id", type=str, default=None, help="device id")
    args = parser.parse_args()

    # Set inputs
    if args.task == "en":
        vocab_path = "../model/vocab_en.txt"
        task_code = 50259
    elif args.task == "zh":
        vocab_path = "../model/vocab_zh.txt"
        task_code = 50260
    else:
        print(
            "\n\033[1;33mCurrently only English or Chinese recognition tasks are supported. Please specify --task as en or zh\033[0m"
        )
        exit(1)
    vocab = read_vocab(vocab_path)


    # Init/Encode/Decode
    encoder_model = init_model(args.encoder_model_path, args.target, args.device_id)
    decoder_model = init_model(args.decoder_model_path, args.target, args.device_id)

    transcribe_live(
        vocab,
        task_code,
        args.input,
        encoder_model,
        decoder_model
    )

    release_model(encoder_model)
    release_model(decoder_model)
