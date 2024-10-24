# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import WhisperFeatureExtractor
import gradio as gr

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--port", default=9527)
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

# gradio 
def gradio_reset(chat_state):
    
    chat_state = []
    return (None,
            gr.update(value=None, interactive=True),
            gr.update(placeholder='Please upload your wav first', interactive=False),
            gr.update(value="Upload & Start Chat", interactive=True),
            chat_state)

def upload_speech(gr_speech, text_input, chat_state):
    
    if gr_speech is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state.append(gr_speech)
    return (gr.update(interactive=False),
            gr.update(interactive=True, placeholder='Type and press Enter'),
            gr.update(value="Start Chatting", interactive=False),
            chat_state)

def gradio_ask(user_message, chatbot, chat_state):
    
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state.append(user_message)
    chatbot.append([user_message, None])
    # 
    return gr.update(interactive=False, placeholder='Currently only single round conversations are supported.'), chatbot, chat_state

def gradio_answer(chatbot, chat_state, num_beams, temperature, top_p):
    samples = prepare_one_sample(chat_state[0], wav_processor)
    prompt = [
        cfg.config.model.prompt_template.format(chat_state[1].strip())
    ]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        llm_message = model.generate(
            samples, cfg.config.generate, prompts=prompt
        )
    chatbot[-1][1] = llm_message[0]
    return chatbot, chat_state

title = """<h1 align="center">SALMONN: Speech Audio Language Music Open Neural Network</h1>"""
image_src = """<h1 align="center"><a href="https://github.com/bytedance/SALMONN"><img src="https://raw.githubusercontent.com/bytedance/SALMONN/main/resource/salmon.png", alt="SALMONN" border="0" style="margin: 0 auto; height: 200px;" /></a> </h1>"""
description = """<h3>This is the demo of SALMONN. Upload your audio and start chatting!</h3>"""


with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(image_src)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            speech = gr.Audio(label="Audio", type='filepath')
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="beam search numbers",
            )

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.1,
                interactive=True,
                label="top p",
            )

            temperature = gr.Slider(
                minimum=0.8,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=False,
                label="temperature",
            )

        with gr.Column():
            chat_state = gr.State([])
            
            chatbot = gr.Chatbot(label='SALMONN')
            text_input = gr.Textbox(label='User', placeholder='Please upload your speech first', interactive=False)

    with gr.Row():
        examples = gr.Examples(
            examples = [
                ["resource/audio_demo/gunshots.wav", "Recognize the speech and give me the transcription."],
                ["resource/audio_demo/gunshots.wav", "Provide the phonetic transcription for the speech."],
                ["resource/audio_demo/gunshots.wav", "Please describe the audio."],
                ["resource/audio_demo/gunshots.wav", "Recognize what the speaker says and describe the background audio at the same time."],
                ["resource/audio_demo/gunshots.wav", "Please answer the speaker's question in detail based on the background sound."],
                ["resource/audio_demo/duck.wav", "Please list each event in the audio in order."],
                ["resource/audio_demo/duck.wav", "Based on the audio, write a story in detail. Your story should be highly related to the audio."],
                ["resource/audio_demo/duck.wav", "How many speakers did you hear in this audio? Who are they?"],
                ["resource/audio_demo/excitement.wav", "Describe the emotion of the speaker."],
                ["resource/audio_demo/mountain.wav", "Please answer the question in detail."],
                ["resource/audio_demo/music.wav", "Please describe the music in detail."],
                ["resource/audio_demo/music.wav", "What is the emotion of the music? Explain the reason in detail."],
                ["resource/audio_demo/music.wav", "Can you write some lyrics of the song?"],
                ["resource/audio_demo/music.wav", "Give me a title of the music based on its rhythm and emotion."]
            ],
            inputs=[speech, text_input]
        )
        
    upload_button.click(upload_speech, [speech, text_input, chat_state], [speech, text_input, upload_button, chat_state])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state], [chatbot, chat_state]
    )
    clear.click(gradio_reset, [chat_state], [chatbot, speech, text_input, upload_button, chat_state], queue=False)



# demo.launch(share=True, enable_queue=True, server_port=int(args.port))
demo.launch(share=True, server_port=int(args.port))
