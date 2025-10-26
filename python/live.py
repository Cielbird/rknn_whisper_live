"""
Module for the live implementation of ONNX/RKNN whisper
"""

import time
import sounddevice as sd
import numpy as np
import soundfile as sf

import onnxruntime

from audio_utils import FakeInputStream, log_mel_spectrogram, pad_or_trim
from util import base64_decode, detect_any_repetition_loop

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_LENGTH = CHUNK_LENGTH * 100  # CHUNK_LENGTH * SAMPLE_RATE / HOP_LENGTH
N_MELS = 80
TRANSCRIBE_INTERVAL = 5  # wait at least 5 seconds between each transcription
# used to detect loops in the decoding step: kill the decoding
TOKEN_LOOP_MAX_LEN, TOKEN_LOOP_MIN_REPS = 10, 4


# from rknn.api import RKNN
# comment this when using RKNN:
# pylint: disable=missing-class-docstring
class RKNN:
    # pylint: disable=unused-argument, missing-function-docstring
    def load_rknn(self, model):
        return 0

    # pylint: disable=unused-argument, missing-function-docstring
    def init_runtime(self, target, device_id):
        return 0


def init_model(model_path, target=None, device_id=None):
    """Init a ONNX or RKNN model with a path to the model file"""
    if model_path.endswith(".rknn"):
        # Create RKNN object
        model = RKNN()

        # Load RKNN model
        print("--> Loading model")
        ret = model.load_rknn(model_path)
        if ret != 0:
            print(f'Load RKNN model "{model_path}" failed!')
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
    """Clean up a ONNX or RKNN model"""
    if "rknn" in str(type(model)):
        model.release()
    elif "onnx" in str(type(model)):
        del model
    model = None

class LiveWhisper:
    """
    Class responsible for live static whisper implementation:
    chunking audio, dispatching the Whisper model, and merging outputs
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        target: str,
        device_id: str | None,
        vocab: dict,
    ):
        self.encoder_model = init_model(encoder_path, target, device_id)
        self.decoder_model = init_model(decoder_path, target, device_id)
        self.vocab = vocab
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_offset = 0
        self.results_buffer = []
        self.debug_counter = 0

    def run(self, task_code: int, audio_file: str | None):
        """
        Live transcription loop
        """
        if audio_file is None:
            stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=1024,
                callback=self.audio_callback,
            )
        else:
            stream = FakeInputStream(audio_file, callback=self.audio_callback)

        with stream:
            print("Recording... press Ctrl+C to stop.")
            last_time = time.time()

            while True:
                now = time.time()
                if now - last_time >= TRANSCRIBE_INTERVAL:
                    last_time = now
                    # make a copy of the buffer for thread safety
                    data_copy = np.copy(self.audio_buffer)
                    self.process_audio(data_copy, task_code)

    def audio_callback(self, indata, _frames, _time_info, status):
        """Callback that appends new audio to the current buffer"""
        if status:
            print("Status:", status)

        # take first channel
        indata = indata[:, 0]
        # append new chunk
        self.audio_buffer = np.append(self.audio_buffer, indata)
        # clip
        clip_start = max(0, len(self.audio_buffer) - N_SAMPLES)
        self.buffer_offset += clip_start
        self.audio_buffer = self.audio_buffer[clip_start:-1]

    def process_audio(self, data: np.array, task_code):
        # Try to transcribe from buffer
        audio_array = log_mel_spectrogram(data, N_MELS, N_FFT, HOP_LENGTH).numpy()
        x_mel = pad_or_trim(audio_array, N_MELS, MAX_LENGTH)

        self.debug_counter += 1
        sf.write(f"logs/log_{self.debug_counter}.wav", data, SAMPLE_RATE)
        print(f"==== Frame: {self.debug_counter}")

        x_mel = np.expand_dims(x_mel, 0)

        out_encoder = self.run_encoder(x_mel)
        result = self.run_decoder(out_encoder, task_code)
        result_str = self.tokens_to_text(result, task_code)
        print("Whisper output:", result_str)

        # TODO this was used to merge results: repurpose this
        # audio_start = self.buffer_offset / SAMPLE_RATE
        # audio_end = (self.buffer_offset + len(self.audio_buffer)) / SAMPLE_RATE
        # self.results_buffer.append((audio_start, audio_end, result))
        # merged = merge_segments(self.results_buffer)
        # print("Merged:", merged)

    def run_encoder(self, in_encoder) -> np.ndarray:
        """
        Run the encoder model

        Return:
            np.ndarray with shape (1, 1000, 512)
            (with chunk size 10s)
        """
        out_encoder = None
        if "rknn" in str(type(self.encoder_model)):
            out_encoder = self.encoder_model.inference(inputs=in_encoder)[0]
        elif "onnx" in str(type(self.encoder_model)):
            out_encoder = self.encoder_model.run(None, {"x": in_encoder})[0]

        return out_encoder

    def _decode(self, tokens: list[int], out_encoder: np.ndarray) -> np.ndarray:
        out_decoder = None
        if "rknn" in str(type(self.decoder_model)):
            out_decoder = self.decoder_model.inference(
                [np.asarray([tokens], dtype="int64"), out_encoder]
            )[0]
        elif "onnx" in str(type(self.decoder_model)):
            out_decoder = self.decoder_model.run(
                None,
                {"tokens": np.asarray([tokens], dtype="int64"), "audio": out_encoder},
            )[0]

        return out_decoder

    def run_decoder(self, out_encoder: np.ndarray, task_code: int) -> list[int]:
        """Execute decoder model"""
        # tokenizer = whisper.decoding.get_tokenizer( True, #model.is_multilingual
        #                                             task="transcribe",
        #                                             language="en",
        #                                             )

        end_token = 50257  # tokenizer.eot
        next_token = 50258  # tokenizer.sot
        transcribe_token = 50359
        _notimestamps_token = 50363
        # no notimestamps token: lets see
        timestamp_begin = 50364  # tokenizer.timestamp_begin

        max_tokens = 48
        # the size 48 token buffer passed to the static decoder model
        tokens = [
            next_token,
            task_code,
            transcribe_token,
            timestamp_begin, # use notimestamps_token at the end to get timestamps
        ]
        preamble_len = len(tokens)
        tokens = tokens * int(max_tokens / preamble_len)

        # pop old tokens when buffer is full, except for the first tokens in the preamble
        pop_id = max_tokens

        # final generated tokens list
        final_tokens = []

        while next_token != end_token:
            out_decoder = self._decode(tokens, out_encoder)
            next_token = out_decoder[0, -1].argmax()
            tokens.append(next_token)

            if detect_any_repetition_loop(tokens, TOKEN_LOOP_MAX_LEN, TOKEN_LOOP_MIN_REPS):
                # decoder probablly stuck in loop
                result = self.tokens_to_text(final_tokens, task_code)
                print(f"=== Decoder fucked! : {result}")
                return ""

            if next_token == end_token:
                tokens.pop(-1)
                next_token = tokens[-1]
                break
            if next_token > timestamp_begin:
                print("timestamp detected:", next_token)
                # continue
            if pop_id > preamble_len:
                pop_id -= 1
            else:
                print("=== Buffer full")

            tokens.pop(pop_id)
            final_tokens.append(next_token)

        return final_tokens

    def tokens_to_text(self, tokens: list[int], task_code: int) -> str:
        """
        Convert tokens to text. `task_code` is the language identifying token
        """
        tokens_str = ""
        for token in tokens:
            tokens_str += self.vocab[str(token)]
        result = (
            tokens_str.replace("\u0120", " ")
            .replace("<|endoftext|>", "")
            .replace("\n", "")
        )
        if task_code == 50260:  # TASK_FOR_ZH
            result = base64_decode(result)
        return result
        

    def __del__(self):
        release_model(self.encoder_model)
        release_model(self.decoder_model)
