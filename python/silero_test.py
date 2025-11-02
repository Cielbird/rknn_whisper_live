import numpy as np
import soundfile as sf
import onnxruntime as ort
import matplotlib.pyplot as plt

# --- CONFIG ---
ONNX_MODEL_PATH = "../model/silero_vad.onnx"
AUDIO_FILE = "../model/alex_gui_vad_bilingual.wav"
SAMPLE_RATE = 16000
THRESHOLD = 0.3
CHUNK_MS = 20  # 20 ms chunks

# --- LOAD AUDIO ---
waveform, sr = sf.read(AUDIO_FILE, dtype="float32")
# TODO use ensure_channels etc
waveform = waveform / np.max(np.abs(waveform))  # normalize
audio = np.expand_dims(waveform, axis=0)  # (1, num_samples)

# --- LOAD MODEL ---
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
state = np.zeros((2, 1, 128), dtype=np.float32)
sr_value = np.array(SAMPLE_RATE, dtype=np.int64)

# --- STREAMING INFERENCE ---
chunk_len = int(SAMPLE_RATE * CHUNK_MS / 1000)
num_chunks = (audio.shape[1] + chunk_len - 1) // chunk_len
probs = []

for i in range(num_chunks):
    start = i * chunk_len
    end = min(start + chunk_len, audio.shape[1])
    chunk = audio[:, start:end]

    # pad last chunk
    if chunk.shape[1] < chunk_len:
        pad_width = chunk_len - chunk.shape[1]
        chunk = np.pad(chunk, ((0,0),(0,pad_width)), mode='constant')

    ort_inputs = {'input': chunk, 'state': state, 'sr': sr_value}
    out, state = session.run(None, ort_inputs)
    probs.append(out[0,0])  # single probability

# --- POSTPROCESS ---
probs = np.array(probs)
time_axis = np.arange(len(probs)) * (CHUNK_MS / 1000)
speech_mask = probs > THRESHOLD

def expand_speech_mask(speech_mask: np.ndarray, pre_speech_ms: float, post_speech_ms: float, sample_rate: int) -> np.ndarray:
    n = len(speech_mask)
    pre_samples = int(pre_speech_ms * sample_rate / CHUNK_MS / 1000)
    post_samples = int(post_speech_ms * sample_rate / CHUNK_MS / 1000)

    # Pad to avoid shrinking edges
    padded_mask = np.pad(speech_mask.astype(int), (pre_samples, pre_samples + post_samples + 1), mode='constant')
    kernel = np.ones(pre_samples + post_samples + 1, dtype=int)
    padded_mask = np.convolve(padded_mask, kernel, mode='valid')
    padded_mask = padded_mask[pre_samples:pre_samples + n] > 0
    return padded_mask

speech_mask = expand_speech_mask(speech_mask, 5, 200, SAMPLE_RATE)

time_axis_probs = np.arange(len(probs)) * (CHUNK_MS / 1000)
time_axis_audio = np.linspace(0, len(waveform)/SAMPLE_RATE, len(waveform))

# --- PLOT ---
plt.figure(figsize=(14,5))
plt.plot(time_axis_audio, waveform, alpha=0.6, label="Audio waveform")
plt.plot(time_axis_probs, probs, color='orange', linewidth=2, label="VAD probability")
plt.plot(time_axis_probs, speech_mask.astype(float), color='red', linestyle='--', label="Speech mask")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude / Probability")
plt.title("Audio waveform with Silero VAD overlay")
plt.legend()
plt.grid(True)
plt.show()
