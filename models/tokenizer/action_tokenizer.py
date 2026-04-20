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
# Adapted from https://github.com/openvla/openvla. The original license is located at 'third-party-license/OpenVLA.txt'.

from typing import List, Union
import numpy as np
from transformers import PreTrainedTokenizerBase
import torch

class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.last_vocab_idx = self.tokenizer.pad_token_id - 1

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.last_vocab_idx - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        return list(self.last_vocab_idx - discretized_action)
        # Handle single element vs. batch
        # if len(discretized_action.shape) == 1:
        #     return self.tokenizer.decode(list(self.last_vocab_idx - discretized_action))
        # else:
        #     return self.tokenizer.batch_decode((self.last_vocab_idx - discretized_action).tolist())
    
    def decode_ids_to_text(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs to text using the tokenizer.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            str: Decoded text corresponding to the input token IDs.
        """
        return self.tokenizer.decode(ids)

    def decode_text_to_ids(self, text: str) -> List[int]:
        """
        Encodes a text string back into token IDs using the tokenizer.

        Args:
            text (str): Input text to encode.

        Returns:
            List[int]: Encoded token IDs corresponding to the input text.
        """
        return self.tokenizer.encode(text)

    def encode_actions_to_ids(self, actions: np.ndarray) -> np.ndarray:
        """
        Encodes continuous actions into discrete token IDs.

        Args:
            actions (np.ndarray): Continuous actions to be encoded.

        Returns:
            np.ndarray: Encoded token IDs corresponding to the input actions.
        """
        # Clip the actions within the allowed range
        actions = np.clip(actions, a_min=self.min_action, a_max=self.max_action)
        
        # Discretize the actions into bin indices
        discretized_actions = np.digitize(actions, self.bins) - 1  # digitize returns [1, len(bins)], shift to [0, len(bins)-1]

        # Ensure no index exceeds the bounds of bin_centers
        discretized_actions = np.clip(discretized_actions, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        # Convert discretized actions into token IDs
        action_token_ids = self.last_vocab_idx - discretized_actions

        return action_token_ids

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.last_vocab_idx - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    def decode_text_to_actions(self, text_list: List[str]) -> torch.Tensor:
        """
        Decodes a list of action strings back into continuous actions.

        Args:
            text_list (List[str]): List of action strings.

        Returns:
            torch.Tensor: Decoded continuous actions corresponding to the input text strings.
        """
        # Convert the list of strings to token IDs
        token_ids = [self.tokenizer.encode(text)[0] for text in text_list]  # Assuming one token per string
        
        # Decode the token IDs to continuous actions and convert to torch tensor
        actions = self.decode_token_ids_to_actions(np.array(token_ids))
        return torch.tensor(actions, dtype=torch.float64)


    @property
    def vocab_size(self) -> int:
        return self.n_bins
