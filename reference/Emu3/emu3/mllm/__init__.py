# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.
# Adapted from https://github.com/baaivision/Emu3. The original license is located at 'third-party-license/Emu.txt'.

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_emu3": ["Emu3Config", "Emu3MoEConfig"],
    "tokenization_emu3": ["Emu3Tokenizer"],
    "processing_emu3": ["Emu3Processor"]
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_emu3_mix"] = [
        "Emu3Model",
        "Emu3PretrainedModel",
        "Emu3ForCausalLM",
        "Emu3MoE",
        "Emu3MoEWithSpeech",
        "Emu3ForMix",
        "Emu3ForMix_FourExpert",
        "Emu3ForMix_FourExpert_Text",
        "LlamaWithSpeech"
    ]

if TYPE_CHECKING:
    from .configuration_emu3 import Emu3Config, Emu3MoEConfig
    from .tokenization_emu3 import Emu3Tokenizer
    from .processing_emu3 import Emu3Processor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_emu3_mix import (
            Emu3Model,
            Emu3PretrainedModel,
            Emu3ForCausalLM,
            Emu3MoE,
            Emu3MoEWithSpeech,
            Emu3ForMix,
            Emu3ForMix_FourExpert,
            Emu3ForMix_FourExpert_Text,
            LlamaWithSpeech
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
