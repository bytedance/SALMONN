# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.

import dataclasses
from enum import auto, Enum
from typing import List, Any, Union
import re
import base64
from io import BytesIO
from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()
    R1 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if self.system == "" else self.system + self.sep + "\n"
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret

        elif self.sep_style == SeparatorStyle.LLAMA_3:
            chat_template_messages = [{"role": "system", "content": self.system}]
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, images = message
                        message = "<image>" * len(images) + message
                    chat_template_messages.append({"role": role, "content": message})

            # print(chat_template_messages)
            return self.tokenizer.apply_chat_template(chat_template_messages, tokenize=False, add_generation_prompt=True)
            # ret = "" if self.system == "" else self.system + self.sep + "\n"
            # for role, message in messages:
            #     if message:
            #         if type(message) is tuple:
            #             message, images = message
            #             message = "<image>" * len(images) + message
            #         ret += role + "\n" + message + self.sep + "\n"
            #     else:
            #         ret += role + "\n"
            # return ret

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.GEMMA:
            ret = ""
            for i, (role, message) in enumerate(messages):
                assert role == self.roles[i % 2], "Conversation should alternate user/assistant/user/assistant/..."
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def process_image(self, image, image_process_mode, return_pil=False, image_format="PNG"):
        if image_process_mode == "Pad":

            def expand2square(pil_img, background_color=(122, 116, 104)):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image)
        elif image_process_mode in ["Default", "Crop"]:
            pass
        elif image_process_mode == "Resize":
            image = image.resize((336, 336))
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

        if type(image) is not Image.Image:
            image = Image.open(image).convert("RGB")

        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        max_len, min_len = 672, 448
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
        if return_pil:
            return image
        else:
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()
            return img_b64_str

    def get_images(self, return_pil=False, return_path=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    for img in image:
                        if not return_path:
                            img = self.process_image(img, image_process_mode, return_pil=return_pil)
                        else:
                            images.append(img)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    if type(image) != list:
                        image = [image]
                    if len(image) == 1:
                        msg = "<image>\n" + msg.replace("<image>", "").strip()
                    else:
                        msg = re.sub(r"(<image>)\n(?=<image>)", r"\1 ", msg)
                    for img in image:
                        img_b64_str = self.process_image(img, "Default", return_pil=False, image_format="JPEG")
                        img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}"/>'
                        msg = msg.replace("<image>", img_str, 1).strip()
                    if len(msg) > 0:
                        ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(system=self.system, roles=self.roles, messages=[[x, y] for x, y in self.messages], offset=self.offset, sep_style=self.sep_style, sep=self.sep, sep2=self.sep2, version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

default_conversation = conv_qwen
conv_templates = {
    "default": default_conversation,
    "qwen_1_5": conv_qwen,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
