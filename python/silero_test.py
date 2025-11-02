import soundfile as sf

from silero import SileroVAD

ONNX_MODEL_PATH = "../model/silero_vad.onnx"

# AUDIO_FILE = "../model/test_en.wav"
# AUDIO_FILE = "../model/test_zh.wav"
# AUDIO_FILE = "../model/alex_gui_vad_bilingual.wav"
# AUDIO_FILE = "../model/guilhem_test.wav"
# AUDIO_FILE = "../model/gui_60s.wav"

vad = SileroVAD(model_path=ONNX_MODEL_PATH, target="rk3588", device_id=None)

waveform, sr = sf.read(AUDIO_FILE, dtype="float32")

vad.run(waveform, discard_last=False)
