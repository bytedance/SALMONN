# ELLSA 2 — Standalone Generator Demo

This page demonstrates the **Standalone generator** released with *ELLSA 2: Full-Duplex Interaction with Speech, Expressions and Gestures*. It animates the target person's body, hands, and face from aligned target/partner audio. Upload the two audio tracks and click **Generate video**. The page starts playback when the first video segment is ready, while later motion is being generated and rendered.

## Start

Install the main [release environment](../README.md#installation) and EGL/OpenGL. FFmpeg and ffprobe must be on `PATH`. For hardware rendering, the container needs the NVIDIA OpenGL/EGL userspace libraries matching its host driver. Include `graphics` alongside `compute,utility` in [`NVIDIA_DRIVER_CAPABILITIES`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities) when starting a container. The startup log prints the actual EGL renderer; `llvmpipe` means software rendering. Select a visible EGL device with `EGL_DEVICE_ID` when multiple GPUs are exposed; its index is independent of CUDA device numbering. Obtain the separately licensed SMPL-X neutral 2020 model with 100 expression coefficients using the [asset download guide](../../checkpoints/README.md#external-assets-sources-and-download-guide).

All commands below run from `released/code/`:

```bash
# Install the GPL build to provide libx264 and AAC encoding.
conda install -c conda-forge 'ffmpeg>=6=gpl_*' --freeze-installed -y
python -m pip install -r demo/requirements.txt
python -m pip check
python demo/app.py \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz
```

Open **http://127.0.0.1:7860** after initialization finishes.

Alternatively, add `--share` to create an externally accessible Gradio URL without SSH forwarding. The default server binds to localhost. Run `python demo/app.py --help` for the available options.

## Live microphone

Open **Animate from your microphone**, choose a **Motion prompt prefix**, click **Record**, and speak. The browser captures continuously on an AudioWorklet and sends ordered PCM over a WebSocket, independently of page updates and inference; the generator emits motion in 0.8-second windows with a silent partner. Click **Stop** to flush the final partial window and download the recording, motion NPZ, and video. The live preview is muted to avoid microphone feedback. When generation finishes, the same player switches to the complete video with your voice; click Play to replay it with sound. The downloaded video also includes your voice. Each recording is limited by `--max-seconds` (20 minutes by default).

The microphone offers P1303 (balanced, default), P0092 (moderate), P0421 (more active), P4104 (subtle), and P1305A (previous default). The new references were chosen using speaker-level source-motion quality, valid poses and speaking/listening coverage. P1303 and P0092 appear in the final 81-speaker training inventory; P0421 and P4104 are held-out reference alternatives. Descriptions refer to source motion and do not guarantee generated quality. Each reference contains speaking and listening motion of the same person. The motion-only prefix guides style, without prefix audio or speaking/listening labels, matching the existing conditioning format. Select before recording; each request keeps its chosen prefix through context refresh. The downloads include `motion_prefix.json` to identify the reference used. Inference does not update model weights.

Microphone capture needs browser permission and a localhost or HTTPS page. For a remote server, use localhost SSH forwarding or the HTTPS URL created by `--share`. Input at common microphone sample rates is continuously resampled to 16 kHz. Long recordings use the same 12-second context refresh as uploaded audio: retain three completed windows and re-encode from their first window, without older left context.

Live generation requires the released causal upper-body, hand, and face tokenizers. Rasterization uses the same persistent EGL workers and continuous video encoder as uploaded audio. Hardware EGL and sufficient rendering throughput are needed to keep up with recording; check the printed EGL renderer at startup. Software `llvmpipe` may accumulate delay.

## Code

| File | Purpose |
| --- | --- |
| `app.py` | Audio conversion, page controls, and request serialization |
| `engine.py` | Resident models, overlapping token generation and window decoding |
| `microphone.py` | PCM WebSocket, live input queue, continuous resampling and silent partner |
| `microphone_capture.js`, `microphone_processor.js` | Continuous browser capture, ordered transmission and final-packet flush |
| `render.py` | Resident rasterizers, ordered rendering, and MP4/HLS encoding |
| `prefixes.py` | Curated motion references and speaker-quality provenance |
| `smoothing.py` | Upper-body/hand boundary correction without additional future blocks |
| `requirements.txt` | Optional web dependency for the main release environment |
