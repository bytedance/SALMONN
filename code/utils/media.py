import numpy as np
import subprocess

def add_audio_to_video(silent_video_path, audio_path, output_video_path):
    command = [
        'ffmpeg',
        '-y',
        '-i', silent_video_path,
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '64k',
        '-strict', '-2',
        '-af', 'apad',  # Keep every motion frame when audio ends between model blocks.
        '-shortest',
        output_video_path
    ]
    
    subprocess.run(command, check=True)
    print(f"Video with audio generated successfully: {output_video_path}")


def convert_img_to_mp4(input_pattern, output_file, framerate=30):
    command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file,
        '-y' 
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Video conversion successful. Output file: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")
