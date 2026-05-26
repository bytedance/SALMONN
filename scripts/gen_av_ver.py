from moviepy import VideoFileClip
import os
from tqdm import tqdm
import json

def extract_audio_to_wav(video_path, sample_rate=16000):
    """
    从视频文件中提取音频，保存为同名同路径的 .wav 文件，采样率 16kHz

    参数:
        video_path (str): 视频文件的完整路径
        sample_rate (int): 目标采样率，默认为 16000 Hz

    返回:
        str: 生成的音频文件路径
    """
    try:
        # 加载视频文件
        video = VideoFileClip(video_path)
        
        # 提取音频
        audio = video.audio
        if audio is None:
            raise ValueError("视频中没有音频轨道")

        # 构造输出文件路径（同名同路径，扩展名为 .wav）
        base_name = os.path.splitext(video_path)[0]
        output_path = base_name + ".wav"

        # 写入音频文件，设置采样率
        audio.write_audiofile(
            output_path,
            fps=sample_rate,      # 设置采样率
            codec='pcm_s16le',    # 无损 PCM 编码，标准 WAV 格式
            ffmpeg_params=["-ac", "1"]  # 可选：转为单声道，如需立体声请删除此行
        )

        # 释放资源
        video.close()

        print(f"✅ 音频已保存至: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ 提取失败: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 替换为你的视频路径
    with open("/mnt/bn/tiktok-mm-5/aiic/public/data/WorldSense/worldsense_test.json", "r") as f:
        data = json.load(f)

    out_data = []

    for d in data:
        video_file = d["video"]
        # audio_path = extract_audio_to_wav(video_file)
        if not video_file.endswith(".mp4"):
            print("GGGG", video_file)
        audio_path = video_file.replace(".mp4", ".wav")
        d["audio"] = audio_path
        out_data.append(d)

    vid_path = list(set([d["video"] for d in data]))

    for v in tqdm(vid_path):
        extract_audio_to_wav(v)

    with open("/mnt/bn/tiktok-mm-5/aiic/public/data/WorldSense/worldsense_test_av.json", "w") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)