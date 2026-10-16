"""Uploaded dialogue or live microphone -> streaming motion and video rendering."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import math
import os
import re
from pathlib import Path
from threading import Lock
import shutil
import subprocess
import sys
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
from demo.prefixes import DEFAULT_PREFIX, PREFIX_LABELS, load_motion_prefixes

EXAMPLE_IDS = tuple(PREFIX_LABELS)


def run(command, log):
    """Run audio conversion and retain its diagnostics in the request log."""
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True,
                   stdout=log, stderr=subprocess.STDOUT)


def prepare_audio(target, partner, directory, max_seconds, log):
    """Keep both timelines; pad the preview to the model's final 0.8 s block."""
    lengths = []
    for source, name in ((target, "target"), (partner, "partner")):
        output = directory / f"{name}.wav"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", source,
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             "-map_metadata", "-1", output], log)
        with wave.open(str(output)) as audio:
            lengths.append(audio.getnframes())
    if min(lengths) == 0 or max(lengths) > max_seconds * 16000:
        raise ValueError(f"Each audio must contain sound samples and be at most {max_seconds:g} seconds.")
    windows = (max(lengths) + 12799) // 12800
    samples = windows * 12800
    # Left = target, right = partner, including silence for unequal lengths.
    run(["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-i", directory / "target.wav", "-i", directory / "partner.wav",
         "-filter_complex", f"[0:a]apad=whole_len={samples}[a];"
         f"[1:a]apad=whole_len={samples}[b];[a][b]amerge=inputs=2[out]",
         "-map", "[out]", "-t", str(windows * 0.8), "-c:a", "pcm_s16le",
         "-map_metadata", "-1", directory / "dialogue.wav"], log)
    return windows


def generate(target, partner, args, worker=None, engine=None, pose_guide=None, guide_time=3.2, guide_start=0):
    """Prepare one request and keep all GPU work on the resident worker thread."""
    if not target or not partner:
        raise ValueError("Upload both target and partner audio before generating.")
    directory = args.output_dir / uuid.uuid4().hex
    directory.mkdir(parents=True)
    yield None, None, "Preparing audio…"
    with (directory / "run.log").open("w") as log:
        windows = prepare_audio(target, partner, directory, args.max_seconds, log)
    stream = engine.generate(directory, windows, pose_guide, guide_time, guide_start)
    try:
        # Keep CUDA and renderer ownership on one thread, across all requests.
        while (result := worker.submit(next, stream, None).result()) is not None:
            yield result
    finally:
        worker.submit(stream.close).result()


def segment_duration(path, segment_frames=48):
    """Use the fixed video frame count and padded audio timeline, without probing."""
    path = Path(path)
    with wave.open(str(path.parent.parent / "dialogue.wav")) as audio:
        frames = round(audio.getnframes() * 30 / audio.getframerate())
    return min(segment_frames, frames - int(path.stem) * segment_frames) / 30


def build_demo(args, worker, engine):
    """Build upload controls, streaming playback and consistent downloads."""
    import gradio as gr
    from demo.microphone import LiveAudio, install_microphone_routes

    prefixes = load_motion_prefixes()

    live_sessions = {}
    live_lock = Lock()

    def cleanup_microphone(key):
        with live_lock:
            audio = live_sessions.pop(key, None)
        if audio is not None:
            audio.cancel()

    def failed_microphone(key):
        cleanup_microphone(key)
        return gr.update(interactive=True)

    def get_microphone(key):
        with live_lock:
            return live_sessions.get(key)

    @asynccontextmanager
    async def microphone_lifespan(app):
        install_microphone_routes(app, get_microphone)
        yield

    def start_microphone(prefix_id):
        key = uuid.uuid4().hex
        audio = LiveAudio(args.output_dir / key, args.max_seconds)
        audio.prefix_id = prefix_id
        with live_lock:
            live_sessions[key] = audio
        return (key, key, "Connecting microphone…", gr.update(interactive=False),
                gr.Video(value=None, streaming=True, autoplay=True))

    def replay_microphone(files):
        if not files:
            return gr.skip(), gr.update(interactive=True)
        video_path = next(path for path in files if Path(path).suffix == ".mp4")
        return gr.Video(value=video_path, streaming=False, autoplay=False), gr.update(interactive=True)

    def animate_microphone(key):
        with live_lock:
            audio = live_sessions[key]
        stream = engine.generate_microphone(audio, audio.prefix_id)
        try:
            while (result := worker.submit(next, stream, None).result()) is not None:
                yield result
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            raise gr.Error(str(error)) from error
        finally:
            audio.cancel()
            worker.submit(stream.close).result()
            cleanup_microphone(key)

    def submit(target, partner, pose_guide, guide_time, guide_start):
        """Translate request errors into messages displayed by Gradio."""
        try:
            for video, files, status in generate(target, partner, args, worker, engine, pose_guide, guide_time, guide_start):
                yield video, files, status
        except ValueError as error:
            raise gr.Error(str(error)) from error
        except (subprocess.CalledProcessError, RuntimeError) as error:
            raise gr.Error("Generation failed. See run.log in the latest output folder.") from error

    with gr.Blocks(title="ELLSA2 · Dialogue to motion", analytics_enabled=False,
                   delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# ELLSA2 · Dialogue to motion\n"
                    "Upload two aligned audio tracks to animate the **target person’s gestures and expressions**.")
        with gr.Row():
            with gr.Column():
                target = gr.Audio(label="Target person", type="filepath", sources=["upload"], editable=False, buttons=[])
                partner = gr.Audio(label="Dialogue partner", type="filepath", sources=["upload"], editable=False, buttons=[])
                gr.Markdown(f"Start both tracks at the same point in the conversation. "
                            f"Up to **{args.max_seconds / 60:g} min** each; WAV, MP3 or FLAC. "
                            "Shorter tracks and the last animation block are padded with silence.")
                with gr.Accordion("Guide the gestures with a reference motion", open=False, visible=args.motion_bias):
                    guide = gr.File(label="Pose reference (.npz, 30 fps)", file_types=[".npz"], type="filepath")
                    guide_time = gr.Number(label="Insert at (seconds, multiples of 0.8)", value=3.2, minimum=0, step=0.8)
                    guide_start = gr.Number(label="First reference frame", value=0, minimum=0, step=1, precision=0)
                    gr.Markdown("Uses 192 reference frames (6.4 seconds). Facial motion is generated from audio. "
                                "Leave the reference empty for generation without guidance.")
                    gr.Examples(examples=[[str(args.style_path), 3.2, 0]],
                                inputs=[guide, guide_time, guide_start], label="Try the included pose reference")
                button = gr.Button("Generate video", variant="primary")
                status = gr.Markdown("Upload both tracks or choose the example below.")
            with gr.Column():
                video = gr.Video(label="Generated animation", interactive=False, height=480,
                                 streaming=True, autoplay=True, buttons=[])
                motion = gr.File(label="Download motion and video", file_count="multiple",
                                 interactive=False, height=110)
                gr.Markdown("Video audio: **target on the left**, **partner on the right**. "
                            "The download contains body, hand and facial animation at 30 fps. "
                            + ("Playback starts while later motion is being generated and rendered."
                             if args.decode_window_blocks else
                             "Playback begins after motion generation, then continues while rendering."))
        async def stream_segment(value, output_id, first_chunk):
            """Send the existing HLS bytes without re-encoding their audio."""
            # These HLS segments already share one encoder and timeline. Gradio's
            # default adapter re-encodes audio; pass through the existing bytes.
            chunk = None
            if value is not None:
                path = Path(value)
                chunk = {"data": path.read_bytes(), "extension": ".ts",
                         "duration": segment_duration(path, 24 if path.parent.parent.name in live_sessions
                                                       or engine.decode_window_blocks == 1 else 48)}
            return chunk, {"path": output_id, "is_stream": True,
                           "orig_name": "video-stream.mp4", "meta": {"_type": "gradio.FileData"}}

        video.stream_output = stream_segment
        gr.Examples(examples=[[str(DATA_ROOT / f"testset/audio/{clip_id}_{role}.wav")
                               for role in ("target", "partner")] for clip_id in EXAMPLE_IDS],
                    inputs=[target, partner], example_labels=list(EXAMPLE_IDS),
                    label="Try a conversation", cache_examples=False)
        with gr.Accordion("Animate from your microphone", open=False):
            microphone_key = gr.State(None, delete_callback=cleanup_microphone)
            gr.Markdown(f"Start recording and speak to animate the target person. The partner stays silent. "
                        f"Up to **{args.max_seconds / 60:g} min** per recording. "
                        "The live preview is muted to avoid microphone feedback. After generation finishes, "
                        "replay here or download the video to hear your voice.")
            with gr.Row():
                with gr.Column():
                    motion_prefix = gr.Dropdown(
                        choices=[(prefix["label"], key) for key, prefix in prefixes.items()],
                        value=DEFAULT_PREFIX, label="Motion prompt prefix", elem_id="microphone-prefix")
                    gr.Markdown("Choose before recording to compare movement styles. Each reference includes "
                                "speaking and listening motion from the same person. ")
                    microphone_client_key = gr.Textbox(visible=False)
                    with gr.Row():
                        microphone_record = gr.Button("Record", variant="primary")
                        microphone_stop = gr.Button("Stop")
                    microphone_input_status = gr.Markdown("Start recording to animate your voice.",
                                                         elem_id="microphone-capture-status")
                with gr.Column():
                    live_video = gr.Video(label="Live animation", streaming=True, autoplay=True,
                                          interactive=False, height=480, buttons=[], elem_id="microphone-animation")
                    live_files = gr.File(label="Download recording, motion and video", file_count="multiple")
                    live_status = gr.Markdown("The animation starts while you are recording.")
            live_video.stream_output = stream_segment
        button.click(submit, [target, partner, guide, guide_time, guide_start], [video, motion, status],
                     api_name="generate", concurrency_limit=1, concurrency_id="gpu", show_progress="hidden")
        recording = microphone_record.click(
            start_microphone, inputs=motion_prefix, outputs=[microphone_key, microphone_client_key, microphone_input_status, microphone_record, live_video],
            queue=False, api_name="start_microphone").then(
            fn=None, inputs=microphone_client_key, outputs=microphone_input_status,
            js=Path(__file__).with_name("microphone_capture.js").read_text(), queue=False)
        animation = recording.success(
            animate_microphone, microphone_key, [live_video, live_files, live_status],
            concurrency_limit=1, concurrency_id="gpu", show_progress="hidden", api_name="animate_microphone")
        animation.then(replay_microphone, inputs=live_files, outputs=[live_video, microphone_record],
                       queue=False, api_name=False)
        recording.failure(failed_microphone, inputs=microphone_key, outputs=microphone_record,
                          queue=False, api_name=False)
        microphone_stop.click(fn=None, outputs=microphone_input_status, queue=False,
                              js="async () => window.ellsa2Microphone ? await window.ellsa2Microphone.stop() : 'Start recording first.'")
    demo.microphone_lifespan = microphone_lifespan
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    """Validate assets and launch a single-request resident GPU service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smplx-model", type=Path, required=True,
                        help="Separately licensed SMPLX_NEUTRAL_2020.npz (100D expressions)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rvq-decode-mode", choices=("auto", "stream", "whole"), default="auto")
    parser.add_argument("--decode-window-blocks", type=int, default=1,
                        help="RVQ decode window in 0.8 s blocks; 0 preserves whole-sequence CLI decoding")
    parser.add_argument("--boundary-smoothing-frames", type=int, default=0,
                        help="fade upper-body/hand boundary corrections over 2–24 frames; 0 disables")
    parser.add_argument("--motion-bias", action="store_true", help="load the pose-only guidance checkpoint and enable reference upload")
    parser.add_argument("--style-id", default="fro_00038", help="Reference style from testset/manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/demo")
    parser.add_argument("--max-seconds", type=float, default=1200,
                        help="maximum duration per upload or recording (default: 20 minutes)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create an externally accessible Gradio URL")
    args = parser.parse_args()
    if args.decode_window_blocks < 0:
        parser.error("--decode-window-blocks must be nonnegative")
    if args.boundary_smoothing_frames == 1 or not 0 <= args.boundary_smoothing_frames <= 24:
        parser.error("--boundary-smoothing-frames must be 0 or between 2 and 24")
    if not args.smplx_model.is_file():
        parser.error("--smplx-model must point to an existing SMPL-X model file")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("Install FFmpeg and ffprobe before starting the demo")
    for tool in ("ffmpeg", "ffprobe"):
        version = subprocess.check_output([tool, "-version"], text=True).splitlines()[0]
        major = re.search(r"version n?(\d+)", version)
        if major and int(major[1]) < 6:
            parser.error(f"{tool} 6 or newer is required for streaming; found {version}")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT)
    if not all(re.search(r"\b" + name + r"\b", encoders) for name in ("libx264", "aac")):
        parser.error("FFmpeg must include the libx264 and aac encoders")
    rows = [json.loads(line) for line in (DATA_ROOT / "testset/manifest.jsonl").read_text().splitlines()]
    styles = {row["group_id"]: DATA_ROOT / row["motion_style_pose_path"] for row in rows}
    if args.style_id not in styles:
        parser.error("--style-id must be an ID in testset/manifest.jsonl")
    args.style_path = styles[args.style_id]
    args.smplx_model = args.smplx_model.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.share:
        # Gradio writes its tunnel certificate relative to the working directory.
        os.chdir(args.output_dir)
    os.environ.setdefault("GRADIO_TEMP_DIR", str(args.output_dir / "cache"))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from demo.engine import Engine
    with ThreadPoolExecutor(max_workers=1) as worker:
        engine = worker.submit(Engine, args).result()
        try:
            demo = build_demo(args, worker, engine)
            demo.launch(
                app_kwargs={"lifespan": demo.microphone_lifespan},
                server_name=args.host, server_port=args.port, share=args.share,
                allowed_paths=[str(args.output_dir)], max_file_size="512mb",
                theme="default", footer_links=[])
        finally:
            worker.submit(engine.close).result()


if __name__ == "__main__":
    main()
