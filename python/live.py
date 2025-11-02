"""
Module for the live implementation of ONNX/RKNN whisper
"""

import time

import sounddevice as sd
import numpy as np

from whisper import SAMPLE_RATE, Transcriber, TranscriptionSegment, print_segments

from audio_utils import FakeInputStream
from silero import SileroVAD

TRANSCRIBE_INTERVAL = 5  # wait at least 5 seconds between each transcription
# used to detect loops in the decoding step: kill the decoding
CLIPPING_MAX_SAMPLES = (
    None  # max number of samples in the audio buffer, if none, no max is used
)


class LiveWhisper:
    """
    Class responsible for live static whisper implementation:
    chunking audio, dispatching the Whisper model, and merging outputs
    """

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        vad_path: str,
        target: str,
        device_id: str | None,
        vocab: dict,
    ):
        self.transcriber = Transcriber(
            encoder_path, decoder_path, target, device_id, vocab
        )
        self.vad = SileroVAD(vad_path, target, device_id)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_offset = 0

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
                    self.process_audio(task_code, 0)

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

    def process_audio(self, task_code: int, time_start: float):
        # run VAD on audio
        # pass each chunk individually to transcription
        # clip on >20s, send warning

        audio = np.copy(self.audio_buffer)

        speaking_periods = self.vad.run(audio, discard_last=True)
        if not speaking_periods:
            print(f"VAD clipped {len(audio)/SAMPLE_RATE} seconds")
            return

        last_end_sample = 0
        for period in speaking_periods:
            start_sample = int(period["start"] * SAMPLE_RATE)
            end_sample = int(period["end"] * SAMPLE_RATE)
            print(f"VAD clipped {(start_sample - last_end_sample)/SAMPLE_RATE} seconds")
            last_end_sample = end_sample
            clip = audio[start_sample:end_sample]

            result = self.transcriber.run(clip, task_code)
            if result is not None:
                # add time_start offset
                print("Whisper output:")
                print_segments(result, self.transcriber.vocab, task_code)
        self.audio_buffer = self.audio_buffer[last_end_sample:]
