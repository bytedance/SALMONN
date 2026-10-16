"""Numerical motion comparison and synchronized, labelled comparison videos."""
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from models.vq.config import UPPER_JOINTS


def _rotations(values):
    """Convert flattened axis-angle poses into joint rotations."""
    return Rotation.from_rotvec(values.reshape(-1, 3))


def compare_motion(left_path, right_path):
    """Compare decoders using identical tokens; angular errors are in degrees."""
    with np.load(left_path) as left, np.load(right_path) as right:
        np.testing.assert_array_equal(left["rvq_combined_code_idx"], right["rvq_combined_code_idx"])
        frames = len(left["body_pose"])
        metrics = {"frames": frames, "tokens_identical": True,
                   "decode_window_blocks": int(left["rvq_decode_window_blocks"]),
                   "boundary_smoothing_frames": int(left.get("motion_boundary_smoothing_frames", 0))}
        hands = []
        for key in ("body_pose", "left_hand_pose", "right_hand_pose"):
            a, b = left[key], right[key]
            error = np.rad2deg((_rotations(b).inv() * _rotations(a)).magnitude()).reshape(frames, -1)
            if key == "body_pose":
                error = error[:, UPPER_JOINTS]
                metrics.update(upper_mean_deg=float(error.mean()), upper_p95_deg=float(np.quantile(error, 0.95)),
                               upper_max_deg=float(error.max()))
                # Boundary jumps are measured over the same joint pairs in both decodings.
                boundaries = np.arange(24, frames, 24)
                for label, values in (("stream", a), ("whole", b)):
                    speed = np.rad2deg((_rotations(values[:-1]).inv() * _rotations(values[1:])).magnitude())
                    speed = speed.reshape(frames - 1, -1)[:, UPPER_JOINTS]
                    metrics[f"{label}_boundary_mean_deg"] = float(speed[boundaries - 1].mean()) if len(boundaries) else 0.0
                    metrics[f"{label}_boundary_max_deg"] = float(speed[boundaries - 1].max()) if len(boundaries) else 0.0
            else:
                hands.append(error)
        hands = np.concatenate(hands, axis=1)
        metrics.update(hands_mean_deg=float(hands.mean()), hands_p95_deg=float(np.quantile(hands, 0.95)),
                       jaw_mean_deg=float(np.rad2deg(np.abs(left["motioncode"][:, 103] - right["motioncode"][:, 103])).mean()),
                       expression_mae=float(np.abs(left["motioncode"][:, :100] - right["motioncode"][:, :100]).mean()))
    return metrics


def compose_video(left, right, output, log, label="Zero lookahead (0.8 s)"):
    """Join equal-length videos with one timeline and the original stereo audio."""
    filters = (f"[0:v][1:v]hstack=inputs=2,drawtext=text='{label}':x=12:y=16:fontsize=20:fontcolor=black,"
               "drawtext=text='Whole sequence':x=492:y=16:fontsize=20:fontcolor=black[v]")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(left), "-i", str(right),
                    "-filter_complex", filters, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
                    "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(output)],
                   check=True, stdout=log, stderr=log)


def comparison_files(directory):
    """List streamed/reference motions, their videos and the comparison report."""
    directory = Path(directory)
    names = ["motion.npz", "video.mp4", "whole/motion.npz", "whole/video.mp4", "comparison.mp4", "comparison.json"]
    if (directory / "raw/motion.npz").is_file():
        names.extend(["raw/motion.npz", "raw/video.mp4"])
    return [str(directory / name) for name in names]


def summary_markdown(metrics):
    """Format angular and displayed-mesh differences for visual inspection."""
    layout = ("Left: unsmoothed. Center: smoothed. Right: whole-sequence decoding."
              if "raw_boundary_mean_deg" in metrics else
              "Left: generated-block decoding. Right: whole-sequence decoding.")
    smoothing = (f"Boundary smoothing: {metrics['boundary_smoothing_frames']} frames.\n\n"
                 if metrics.get("boundary_smoothing_frames", 0) else "")
    before_after = (f"| Boundary jump before / after smoothing | {metrics['raw_boundary_mean_deg']:.2f}° / {metrics['stream_boundary_mean_deg']:.2f}° |\n"
                    if "raw_boundary_mean_deg" in metrics else "")
    return (layout + " **Identical tokens.**\n\n" + smoothing
            + "| Difference | Value |\n| --- | --- |\n"
            f"| Upper-body joint rotation, mean / 95th percentile | {metrics['upper_mean_deg']:.2f}° / {metrics['upper_p95_deg']:.2f}° |\n"
            f"| Hand joint rotation, mean / 95th percentile | {metrics['hands_mean_deg']:.2f}° / {metrics['hands_p95_deg']:.2f}° |\n"
            f"| Displayed mesh distance, mean / 95th percentile | {metrics['vertex_mean_mm']:.1f} / {metrics['vertex_p95_mm']:.1f} mm |\n"
            f"| Upper-body jump at 0.8 s boundaries, streamed / whole | {metrics['stream_boundary_mean_deg']:.2f}° / {metrics['whole_boundary_mean_deg']:.2f}° |\n"
            + before_after + "\n"
            + "Pause or replay the paired video to inspect the face and block boundaries. "
            "These are decoder differences, rather than motion-quality scores.")


def add_examples(directory):
    """Add pre-rendered comparisons that can be viewed without starting a GPU job."""
    import json
    import gradio as gr

    directory = Path(directory)
    rows = json.loads((directory / "index.json").read_text())
    cases = {row["label"]: row for row in rows}

    def select(label):
        """Show one synchronized comparison and its numerical report."""
        row = cases[label]
        case = directory / row["directory"]
        return str(case / "comparison.mp4"), summary_markdown(row["metrics"]), comparison_files(case)

    with gr.Accordion("Inspect RVQ decoding and boundary smoothing", open=True):
        selector = gr.Dropdown(list(cases), value=rows[0]["label"], label="Prepared comparison")
        first = select(rows[0]["label"])
        video = gr.Video(value=first[0], label="Synchronized motion comparison", height=480, interactive=False)
        summary = gr.Markdown(first[1])
        files = gr.File(value=first[2], label="Download both motions, videos and metrics", file_count="multiple")
        selector.change(select, selector, [video, summary, files], queue=False, api_name="comparison_example")
