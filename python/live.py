"""
Module for the live implementation of ONNX/RKNN whisper
"""

import time

import sounddevice as sd
import numpy as np

from whisper import SAMPLE_RATE, Transcriber, TranscriptionSegment, print_segments

from audio_utils import FakeInputStream

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
        target: str,
        device_id: str | None,
        vocab: dict,
    ):
        self.transcriber = Transcriber(encoder_path, decoder_path, target, device_id, vocab)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.buffer_offset = 0

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
        result = self.transcriber.run(data, task_code)
        if result is not None:
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
                print_segments(self.confirmed_output, self.transcriber.vocab, task_code)

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
