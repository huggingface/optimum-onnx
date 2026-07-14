# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Small compatibility shims for APIs removed in a Transformers major release."""

from __future__ import annotations

import math
import sys
import types

import torch

from optimum.utils import is_transformers_version


def ensure_transformers_v5_compatibility() -> None:
    """Expose removed v4 helpers still imported by Hub ``trust_remote_code`` models."""
    if not is_transformers_version(">=", "5.0.0"):
        return

    import transformers.pytorch_utils as pytorch_utils
    from transformers import PreTrainedModel

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(
            heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
        ) -> tuple[set[int], torch.LongTensor]:
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head -= sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index: torch.LongTensor = torch.arange(len(mask))[mask].long()
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(pytorch_utils, "prune_conv1d_layer"):

        def prune_conv1d_layer(layer, index: torch.LongTensor, dim: int = 1):
            index = index.to(layer.weight.device)
            weight = layer.weight.index_select(dim, index).detach().clone()
            bias = layer.bias.detach().clone() if dim == 0 else layer.bias[index].detach().clone()
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = pytorch_utils.Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(weight.contiguous())
            new_layer.weight.requires_grad = True
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(bias.contiguous())
            new_layer.bias.requires_grad = True
            return new_layer

        pytorch_utils.prune_conv1d_layer = prune_conv1d_layer

    if "transformers.utils.model_parallel_utils" not in sys.modules:
        model_parallel_utils = types.ModuleType("transformers.utils.model_parallel_utils")

        def assert_device_map(device_map, num_blocks):
            blocks = list(range(num_blocks))
            device_map_blocks = [item for block_list in device_map.values() for item in block_list]
            duplicate_blocks = [
                block for block in dict.fromkeys(device_map_blocks) if device_map_blocks.count(block) > 1
            ]
            missing_blocks = [block for block in blocks if block not in device_map_blocks]
            extra_blocks = [block for block in device_map_blocks if block not in blocks]
            if duplicate_blocks:
                raise ValueError(f"Attention blocks were specified more than once: {duplicate_blocks}")
            if missing_blocks:
                raise ValueError(f"Attention blocks are missing from the device map: {missing_blocks}")
            if extra_blocks:
                raise ValueError(f"The device map contains unknown attention blocks: {extra_blocks}")

        def get_device_map(n_layers, devices):
            layers = list(range(n_layers))
            n_blocks = math.ceil(n_layers / len(devices))
            layers_list = [layers[index : index + n_blocks] for index in range(0, n_layers, n_blocks)]
            return dict(zip(devices, layers_list))

        model_parallel_utils.assert_device_map = assert_device_map
        model_parallel_utils.get_device_map = get_device_map
        sys.modules[model_parallel_utils.__name__] = model_parallel_utils

    if not hasattr(PreTrainedModel, "_convert_head_mask_to_5d"):

        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            if head_mask.dim() != 5:
                raise ValueError(f"head_mask.dim != 5, instead {head_mask.dim()}")
            return head_mask.to(dtype=self.dtype)

        PreTrainedModel._convert_head_mask_to_5d = _convert_head_mask_to_5d

    if not hasattr(PreTrainedModel, "get_head_mask"):

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
                return head_mask
            return [None] * num_hidden_layers

        PreTrainedModel.get_head_mask = get_head_mask
