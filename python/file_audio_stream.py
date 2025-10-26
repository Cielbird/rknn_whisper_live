import time
import soundfile as sf

from util import ensure_channels, ensure_sample_rate

class FakeInputStream:
    def __init__(self, file_path, callback, blocksize=1024):
        self.file_path = file_path
        self.callback = callback
        self.blocksize = blocksize

    def __enter__(self):
        import threading
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.thread.join()

    def _run(self):
        audio_data, sample_rate = sf.read(self.file_path, dtype='float32')
        channels = audio_data.ndim
        audio_data, channels = ensure_channels(audio_data, channels)
        audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
        for i in range(0, len(audio_data), self.blocksize):
            time.sleep(self.blocksize / sample_rate)
            block = audio_data[i:i+self.blocksize]
            self.callback(block.reshape(-1, 1), len(block), None, None)
