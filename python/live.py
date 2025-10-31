"""
Module for the live implementation of ONNX/RKNN whisper
"""

import time
from dataclasses import dataclass

import sounddevice as sd
import numpy as np
import soundfile as sf
import onnxruntime

from audio_utils import FakeInputStream, log_mel_spectrogram, pad_or_trim
from util import base64_decode, detect_any_repetition_loop, token_to_timestamp

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
CLIPPING_MAX_SAMPLES = (
    None  # max number of samples in the audio buffer, if none, no max is used
)
# min gap between timestamp logits to consider a timestamp prediction reliable
MIN_TIMESTAMP_LOGIT = 0.01
# character used to represent a space in the Whisper vocab
VOCAB_SPACE_CHAR = "\u0120"


@dataclass
class TranscriptionSegment:
    """
    A part of a transcription with time bounds
    """

    start: float
    end: float
    tokens: list[int]


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
        self.results_buffer = []  # TODO see if necessary
        self.debug_counter = 0

        # with a LocaAgreement-2 policy
        self.hypothesis_start_offset = 0.0
        self.hypothesis_old = None
        self.hypothesis_new = None
        # list of confirmed transcription segments
        self.confirmed_output = []
        self.confirmed_output_end_time = 0.0
        self.init_time = time.time()

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
                    self.process_audio(data_copy, task_code, self.hypothesis_start_offset)

    def audio_callback(self, indata, _frames, _time_info, status):
        """Callback that appends new audio to the current buffer"""
        if status:
            print("Status:", status)

        # take first channel
        indata = indata[:, 0]
        # append new chunk
        self.audio_buffer = np.append(self.audio_buffer, indata)
        # clip
        if CLIPPING_MAX_SAMPLES:
            clip_start = max(0, len(self.audio_buffer) - CLIPPING_MAX_SAMPLES)
            self.buffer_offset += clip_start
            self.audio_buffer = self.audio_buffer[clip_start:-1]

    def process_audio(self, data: np.array, task_code: int, time_start: float):
        self.debug_counter += 1

        # Try to transcribe from buffer
        audio_array = log_mel_spectrogram(data, N_MELS, N_FFT, HOP_LENGTH).numpy()
        x_mel = pad_or_trim(audio_array, N_MELS, MAX_LENGTH, f"logs/mel_{self.debug_counter}.png")

        sf.write(f"logs/log_{self.debug_counter}.wav", data, SAMPLE_RATE)
        print(f"==== Frame: {self.debug_counter}")

        x_mel = np.expand_dims(x_mel, 0)

        out_encoder = self.run_encoder(x_mel)
        result = self.run_decoder(out_encoder, task_code)
        if result is not None:
            print("Out: ", end="")
            self.print_segments(result, task_code)
            
            # add time_start offset
            for result_segment in result:
                result_segment.start += time_start
                result_segment.end += time_start

            self.hypothesis_old = self.hypothesis_new
            self.hypothesis_new = result

            if self.hypothesis_new is not None and self.hypothesis_old is not None:
                h_old = self.discard_confirmed_output(self.hypothesis_old)
                h_new = self.discard_confirmed_output(self.hypothesis_new)
                new_lcp = self.longest_common_prefix(h_old, h_new)
                if len(new_lcp) > 0:
                    self.confirmed_output.extend(new_lcp)
                    self.confirmed_output_end_time = new_lcp[-1].end
                print("Whisper output:")
                self.print_segments(self.confirmed_output, task_code)

    def longest_common_prefix(self, a: list[TranscriptionSegment], b: list[TranscriptionSegment]):
        """
        Simple implementation of longest common prefix
        """
        lcp = []
        for a_segment, b_segment in zip(a, b):
            if a_segment.tokens != b_segment.tokens:
                break
            lcp.append(a_segment)
        return lcp

    def discard_confirmed_output(self, segments: list[TranscriptionSegment]):
        """
        Discard all segments that end before the confirmed_output_end_time
        """
        return list(filter(lambda s: s.end > self.confirmed_output_end_time, segments))
    
    def print_segments(self, segments: list[TranscriptionSegment], task_code: int):
        for segment in segments:
            tokens_str = self.tokens_to_text(segment.tokens, task_code)
            print(f"{tokens_str}", end="")
        print(" ")


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

    def run_decoder(
        self, out_encoder: np.ndarray, task_code: int
    ) -> list[TranscriptionSegment]:
        """
        Execute whisper decoder model for a chunk of audio

        Args:
        `time_start` - staring time of the audio chunk

        Returns: list of (token timestamp, token id)
        """
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
        token_buffer = [
            next_token,
            task_code,
            transcribe_token,
            timestamp_begin,
        ]
        preamble_len = len(token_buffer)

        # pop old tokens when buffer is full, except for the first tokens in the preamble
        pop_id = max_tokens

        segments = [TranscriptionSegment(0.0, CHUNK_LENGTH, [])]

        while next_token != end_token:
            out_decoder = self._decode(token_buffer, out_encoder)
            logits = out_decoder[0, -1]

            next_token = logits.argmax()

            token_buffer.append(next_token)

            if detect_any_repetition_loop(
                token_buffer, TOKEN_LOOP_MAX_LEN, TOKEN_LOOP_MIN_REPS
            ):
                # decoder probablly stuck in loop
                tokens = []
                for word in segments:
                    tokens.extend(word.tokens)
                _tokens_str = self.tokens_to_text(tokens, task_code)
                print("=== Decoder fucked!")
                return None

            if next_token == end_token:
                token_buffer.pop(-1)
                next_token = token_buffer[-1]
                break

            if next_token > timestamp_begin:
                print("timestamp detected:", next_token)
                # timestamps outputed are 10s off because whisper is made for 30
                # but we are passing 20s!
                timestamp = token_to_timestamp(next_token) - (30 - CHUNK_LENGTH)
                segments[-1].end = timestamp
                segments.append(TranscriptionSegment(timestamp, CHUNK_LENGTH, []))
            else:
                # append to last word
                segments[-1].tokens.append(next_token)

            if pop_id > preamble_len:
                pop_id -= 1
            else:
                pass
                # print(f"=== Buffer full warning")

        # remove empty segments
        segments = [s for s in segments if len(s.tokens) > 0]
        segments = self.split_segments_by_word(segments)
        return segments

    def split_segments_by_word(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        """
        Infer the time bounds of each word in each segment, splitting the segments between words

        Returns a list of segments for each word
        """
        word_segments = []
        for segment in segments:
            if len(segment.tokens) == 0:
                continue
            word_tokens = [[]]
            word_weights = [0]
            for token in segment.tokens:
                if self.vocab[str(token)].startswith(VOCAB_SPACE_CHAR):
                    # new word
                    word_tokens.append([])
                    word_weights.append(0)

                word_tokens[-1].append(token)
                # add the weight (length of token as string)
                word_weights[-1] += len(self.vocab[str(token)])

            # use the word char lens and segment start and end to linearly interpolate for each word
            cummulative_weight = 0
            total_weight = sum(word_weights)
            segment_duration = segment.end - segment.start
            for word, weight in zip(word_tokens, word_weights):
                start = segment.start + (cummulative_weight / total_weight) * segment_duration
                cummulative_weight += weight
                end = segment.start + (cummulative_weight / total_weight) * segment_duration
                word_segments.append(TranscriptionSegment(start, end, word))
        # remove empty segments
        word_segments = [s for s in word_segments if len(s.tokens) > 0]
        return word_segments

    def tokens_to_text(self, tokens: list[int], task_code: int) -> str:
        """
        Convert tokens to text. `task_code` is the language identifying token
        """
        tokens_str = ""
        for token in tokens:
            tokens_str += self.vocab[str(token)]
        result = (
            tokens_str.replace(VOCAB_SPACE_CHAR, " ")
            .replace("<|endoftext|>", "")
            .replace("\n", "")
        )
        if task_code == 50260:  # TASK_FOR_ZH
            result = base64_decode(result)
        return result

    def __del__(self):
        release_model(self.encoder_model)
        release_model(self.decoder_model)
