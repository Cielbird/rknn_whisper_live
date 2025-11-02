"""
Module for running the whisper model with ONNX/RKNN models
"""

from dataclasses import dataclass
import numpy as np
import soundfile as sf
from models import init_model, release_model
from util import base64_decode, detect_any_repetition_loop, token_to_timestamp
from audio_utils import log_mel_spectrogram, pad_or_trim

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 20
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_LENGTH = CHUNK_LENGTH * 100  # CHUNK_LENGTH * SAMPLE_RATE / HOP_LENGTH
N_MELS = 80
TOKEN_LOOP_MAX_LEN, TOKEN_LOOP_MIN_REPS = 10, 4
# character used to represent a space in the Whisper vocab
VOCAB_SPACE_CHAR = "\u0120"

PADDING_TOKEN = 50256
EOT_TOKEN = 50257
SOT_TOKEN = 50258
EN_LANG_TOKEN = 50259
ZH_LANG_TOKEN = 50260
TRANSCRIBE_TASK_TOKEN = 50359
NO_TIMESTAMPS_TOKEN = 50363
FIRST_TIMESTAMP_TOKEN = 50364


@dataclass
class TranscriptionSegment:
    """
    A part of a transcription with time bounds
    """

    start: float
    end: float
    tokens: list[int]


class Transcriber:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        target: str,
        device_id: str | None,
        vocab: dict,
    ):
        self.debug_counter = 0
        self.encoder_model = init_model(encoder_path, target, device_id)
        self.decoder_model = init_model(decoder_path, target, device_id)
        self.vocab = vocab

    def __del__(self):
        release_model(self.encoder_model)
        release_model(self.decoder_model)

    def run(
        self,
        x_audio: np.ndarray,
        task_code: int,
    ) -> list[TranscriptionSegment]:
        """
        Run the entire Whisper transcription on a chunk of audio
        """
        self.debug_counter += 1

        audio_array = log_mel_spectrogram(x_audio, N_MELS, N_FFT, HOP_LENGTH).numpy()
        x_mel = pad_or_trim(
            audio_array, N_MELS, MAX_LENGTH, f"logs/mel_{self.debug_counter}.png"
        )

        sf.write(f"logs/log_{self.debug_counter}.wav", x_audio, SAMPLE_RATE)
        print(f"==== Frame: {self.debug_counter}")

        x_mel = np.expand_dims(x_mel, 0)

        out_encoder = self.run_encoder(x_mel)
        result = self.run_decoder(out_encoder, task_code)
        print_segments(result, self.vocab, task_code)
        return result

    def run_encoder(self, in_encoder: np.ndarray) -> np.ndarray:
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
        self, out_encoder: np.ndarray, lang_code: int
    ) -> list[TranscriptionSegment]:
        """
        Execute whisper decoder model for a chunk of audio

        Args:
        `time_start` - staring time of the audio chunk

        Returns: list of (token timestamp, token id)
        """
        next_token = SOT_TOKEN  # tokenizer.sot

        max_tokens = 48
        # the size 48 token buffer passed to the static decoder model
        token_buffer = [
            SOT_TOKEN,
            lang_code,
            TRANSCRIBE_TASK_TOKEN,
            FIRST_TIMESTAMP_TOKEN,  # we want timestamps
        ]
        preamble_len = len(token_buffer)
        token_buffer.extend([PADDING_TOKEN] * (max_tokens - preamble_len))

        insert_idx = preamble_len

        segments = [TranscriptionSegment(0.0, CHUNK_LENGTH, [])]

        while next_token != EOT_TOKEN:
            # pad token buffer with padding tokens
            out_decoder = self._decode(token_buffer, out_encoder)

            logits = out_decoder[0, insert_idx - 1]
            next_token = logits.argmax()

            token_buffer.insert(insert_idx, next_token)
            if insert_idx >= max_tokens:
                # remove oldest token, skipping preamble
                print("Warning: buffer full")
                token_buffer.pop(preamble_len - 1)
            else:
                token_buffer.pop()
                insert_idx += 1

            if next_token == EOT_TOKEN:
                break

            if detect_any_repetition_loop(token_buffer[:insert_idx], 8, 3):
                print("Error: decoder stuck")
                break

            timestamp = token_to_timestamp(next_token)
            if timestamp is not None:
                print("timestamp detected:", self.vocab[str(next_token)])
                # timestamps outputed are 10s off because whisper is made for 30
                # but we are passing 20s!
                timestamp = timestamp - (30 - CHUNK_LENGTH)
                segments[-1].end = timestamp
                segments.append(TranscriptionSegment(timestamp, CHUNK_LENGTH, []))
            else:
                # append to last word
                segments[-1].tokens.append(next_token)

        # remove empty segments
        segments = [s for s in segments if len(s.tokens) > 0]
        segments = split_segments_by_word(segments, self.vocab)
        return segments


def tokens_to_text(tokens: list[int], vocab: dict, task_code: int) -> str:
    """
    Convert tokens to text. `task_code` is the language identifying token
    """
    tokens_str = ""
    for token in tokens:
        tokens_str += vocab[str(token)]
    result = (
        tokens_str.replace(VOCAB_SPACE_CHAR, " ")
        .replace("<|endoftext|>", "")
        .replace("\n", "")
    )
    if task_code == ZH_LANG_TOKEN:
        result = base64_decode(result)
    return result


def print_segments(segments: list[TranscriptionSegment], vocab: dict, task_code: int):
    """
    Print the text of a series of segments
    """
    for segment in segments:
        tokens_str = tokens_to_text(segment.tokens, vocab, task_code)
        print(f"{tokens_str}", end="")
    print(" ")


def split_segments_by_word(
    segments: list[TranscriptionSegment], vocab: dict
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
            if vocab[str(token)].startswith(VOCAB_SPACE_CHAR):
                # new word
                word_tokens.append([])
                word_weights.append(0)

            word_tokens[-1].append(token)
            # add the weight (length of token as string)
            word_weights[-1] += len(vocab[str(token)])

        # use the word char lens and segment start and end to linearly interpolate for each word
        cummulative_weight = 0
        total_weight = sum(word_weights)
        segment_duration = segment.end - segment.start
        for word, weight in zip(word_tokens, word_weights):
            start = (
                segment.start + (cummulative_weight / total_weight) * segment_duration
            )
            cummulative_weight += weight
            end = segment.start + (cummulative_weight / total_weight) * segment_duration
            word_segments.append(TranscriptionSegment(start, end, word))
    # remove empty segments
    word_segments = [s for s in word_segments if len(s.tokens) > 0]
    return word_segments
