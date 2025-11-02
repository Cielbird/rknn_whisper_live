"""
Module for the Silero VAD model, in ONNX or RKNN format
"""

from matplotlib import pyplot as plt
import numpy as np
from models import init_model, release_model

# sr and chunk size values taken from silero's repository
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
CONTEXT_SAMPLES = 64


THRESHOLD = 0.5  # probability to trigger
PRE_TRIGGER_CONTEXT = 200.0  # ms
TRIGGER_COOLDOWN = 1000  # ms
# how many chunks to cooldown
COOLDOWN_CHUNKS = int((TRIGGER_COOLDOWN / 1000) * SAMPLE_RATE / CHUNK_SAMPLES)


class SileroVAD:
    """
    Class for the Silero VAD model
    """

    def __init__(
        self,
        model_path: str,
        target: str,
        device_id: str | None,
    ):
        self.model = init_model(model_path, target, device_id)

    def __del__(self):
        release_model(self.model)

    def run(self, audio: np.ndarray, discard_last: bool = True):
        """
        Run VAD on an audio, returning the periods of speach in the audio
        - `audio` : 1d audio array, sampled at 16kHz
        - `discard_last` : should the last unfinished speaking period be discarded?
        """
        speaking_periods = []
        probs = []
        audio = np.expand_dims(audio, axis=0)  # (1, num_samples)
        audio_samples = audio.shape[1]
        num_chunks = (audio_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
        is_speaking = False
        context = np.zeros((1, CONTEXT_SAMPLES), dtype="float32")
        state = np.zeros((2, 1, 128), dtype=np.float32)
        for i in range(num_chunks):
            start = i * CHUNK_SAMPLES
            end = min(start + CHUNK_SAMPLES, audio_samples)
            chunk = audio[:, start:end]
            x_audio = np.concat([context, chunk], axis=1)
            context = chunk[:, -CONTEXT_SAMPLES:]

            # pad last chunk
            input_len = CHUNK_SAMPLES + CONTEXT_SAMPLES
            if x_audio.shape[1] < input_len:
                pad_width = input_len - x_audio.shape[1]
                x_audio = np.pad(x_audio, ((0, 0), (0, pad_width)), mode="constant")

            prob, state = self._run(x_audio, state)
            probs.append(prob)  # single probability
            timestamp = i * CHUNK_SAMPLES / SAMPLE_RATE
            if is_speaking:
                if prob < THRESHOLD:
                    is_speaking = False
                    # update speaking periods
                    speaking_periods[-1]["end"] = timestamp + (COOLDOWN_CHUNKS * CHUNK_SAMPLES) / SAMPLE_RATE
            else:
                if prob >= THRESHOLD:
                    is_speaking = True

                    # update speaking periods
                    start = timestamp - PRE_TRIGGER_CONTEXT / 1000
                    # if the previous period ended after the start of this period, we'll just
                    # update the last period upon the end of this one.
                    if not speaking_periods or speaking_periods[-1]["end"] < start:
                        speaking_periods.append(
                            {"start": start, "end": timestamp}
                        )
        audio_time = audio_samples / SAMPLE_RATE
        if is_speaking or (speaking_periods and speaking_periods[-1]["end"] > audio_time):
            if discard_last:
                print("pop")
                speaking_periods.pop()
            else:
                speaking_periods[-1]["end"] = audio_time

        print(len(speaking_periods))
        plot_vad_with_audio(audio, probs, speaking_periods)
        return speaking_periods

    def _run(self, x_audio: np.ndarray, state: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Run the Silero VAD model on a chunk of audio, outputting a probability
        - `x_audio` : audio with shape (1, 576) (512 + context of 64 samples)
        - `state` : state matrix with shape (2, 1, 128)

        The state should be zeros on first call, then output of last call should be used for next
        call.

        Returns
        tuple with values (prob, state)
        - `prob` : probability
        - `state` : state matrix with shape (2, 1, 128)
        """
        assert x_audio.shape == (1, 576)
        assert state.shape == (2, 1, 128)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)

        out = None
        if "rknn" in str(type(self.model)):
            out, state = self.model.inference([x_audio, state, sr])[0]
        elif "onnx" in str(type(self.model)):
            out, state = self.model.run(
                None,
                {"input": x_audio, "state": state, "sr": sr},
            )

        return out[0, 0], state



def plot_vad_with_audio(audio, probs, speaking_periods):
    """
    Plot waveform + VAD probabilities + detected speech regions.
    """
    # Compute time axes
    audio_time = np.arange(audio.shape[1]) / SAMPLE_RATE
    vad_time = np.arange(len(probs)) * (CHUNK_SAMPLES / SAMPLE_RATE)

    # Normalize audio for plotting
    norm_audio = audio[0] / np.max(np.abs(audio))

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # --- Audio waveform ---
    ax1.plot(audio_time, norm_audio, color='gray', alpha=0.6, label='Audio waveform')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude", color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # --- VAD probabilities on a second y-axis ---
    ax2 = ax1.twinx()
    ax2.plot(vad_time, probs, color='blue', label='Speech probability')
    ax2.set_ylabel("Speech Probability", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 1)

    # --- Highlight detected speech segments ---
    for seg in speaking_periods:
        ax1.axvspan(seg["start"], seg["end"], color='orange', alpha=0.3)

    # --- Final touches ---
    fig.suptitle("Silero VAD Output with Audio Overlay")
    fig.tight_layout()
    fig.legend(loc='upper right')
    plt.show()
    