"""Thread-safe microphone input, continuous resampling and aligned silent partner."""
from pathlib import Path
from queue import Queue, Empty
from threading import RLock
import wave

import numpy as np
import soxr


class LiveAudio:
    def __init__(self, directory, max_seconds):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        self.max_seconds = max_seconds
        self.queue = Queue()
        self.lock = RLock()
        self.pending = np.zeros(0, dtype=np.float32)
        self.rate = self.resampler = None
        self.source_frames = self.samples = 0
        self.finished = self.cancelled = False
        self.files = []
        for name, channels in (("target", 1), ("partner", 1), ("dialogue", 2)):
            handle = wave.open(str(self.directory / f"{name}.wav"), "wb")
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            self.files.append(handle)

    def _append(self, audio):
        if not len(audio):
            return
        self.samples += len(audio)
        pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype('<i2')
        silence = np.zeros_like(pcm)
        for handle, values in zip(self.files, (pcm, silence, np.column_stack((pcm, silence)))):
            handle.writeframes(values.tobytes())
        self.pending = np.concatenate((self.pending, audio))
        while len(self.pending) >= 12800:
            block, self.pending = self.pending[:12800].copy(), self.pending[12800:]
            self.queue.put((block, np.zeros_like(block)))

    def feed(self, audio):
        """Accept each browser chunk once; retain the resampler's filter history."""
        with self.lock:
            if audio is None or self.finished:
                return
            rate, values = audio
            if self.rate is None:
                self.rate = rate
                if rate != 16000:
                    self.resampler = soxr.ResampleStream(rate, 16000, 1, dtype='float32')
            if rate != self.rate:
                raise ValueError("Microphone sample rate changed during recording.")
            values = np.asarray(values)
            values = values.astype(np.float32) / 32768 if values.dtype == np.int16 else values.astype(np.float32)
            if values.ndim == 2:
                values = values.mean(axis=1)
            remaining = max(0, round(self.max_seconds * rate) - self.source_frames)
            limit_reached = len(values) >= remaining
            values = values[:remaining]
            self.source_frames += len(values)
            self._append(self.resampler.resample_chunk(values) if self.resampler else values)
            if limit_reached:
                self.finish()

    def finish(self, recording=None):
        """Flush the resampler and pad only the last window."""
        with self.lock:
            if self.finished:
                return
            if recording is not None:
                rate, values = recording
                # Optional complete clip for callers that retain a recording.
                # The live WebSocket already supplies every sample in order.
                if len(values) > self.source_frames:
                    self.feed((rate, values[self.source_frames:]))
                if self.finished:
                    return
            if self.resampler is not None:
                self._append(self.resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True))
            if len(self.pending):
                self._append(np.zeros(12800 - len(self.pending), dtype=np.float32))
            self.finished = True
            for handle in self.files:
                handle.close()
            self.queue.put(None)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if not self.finished:
                self.finished = True
                for handle in self.files:
                    handle.close()
                self.queue.put(None)

    def __iter__(self):
        while True:
            try:
                block = self.queue.get(timeout=30)
            except Empty:
                self.finish()
                block = self.queue.get()
            if self.cancelled or block is None:
                return
            yield block


def install_microphone_routes(app, get_audio):
    """Receive ordered PCM independently of Gradio uploads and inference queues."""
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    @app.get("/microphone/processor.js")
    async def processor():
        return FileResponse(Path(__file__).with_name("microphone_processor.js"), media_type="text/javascript")

    @app.websocket("/microphone/socket/{key}")
    async def microphone_socket(websocket: WebSocket, key: str):
        audio = get_audio(key)
        if audio is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            config = await websocket.receive_json()
            rate = int(config["sample_rate"])
            if rate <= 0:
                raise ValueError("Invalid microphone sample rate.")
            limit_sent = False
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    audio.feed((rate, np.frombuffer(message["bytes"], dtype="<f4")))
                    if audio.finished and not limit_sent:
                        await websocket.send_json({"type": "limit"})
                        limit_sent = True
                elif message.get("text") == "stop":
                    audio.finish()
                    await websocket.send_json({"type": "finished", "source_frames": audio.source_frames})
                    break
                else:
                    raise ValueError("Expected microphone PCM or stop.")
        except WebSocketDisconnect:
            pass
        except (ValueError, KeyError) as error:
            await websocket.send_json({"type": "error", "message": str(error)})
        finally:
            audio.finish()
