# Copyright 2022 The HuggingFace Team. All rights reserved.
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

def debug_torch_export_export(model, dummy_inputs, dynamic_shapes):
    """
    This helps to pinpoint the location of line failing the shape validation.
    This is a temporary solution until something better is implemented.
    The function checks that :func:`torch.export.export` works
    with the given arguments.
    """
    import torch
    from onnx_diagnostic.helpers import string_type
    from onnx_diagnostic.torch_export_patches import torch_export_patches
    from onnx_diagnostic.torch_export_patches.patch_inputs import use_dyn_not_str

    print(f"dynamic_shapes={dynamic_shapes}")
    print(f"dummy_inputs={string_type(dummy_inputs, with_shape=True)}")
    with torch_export_patches(patch_torch=True, patch_transformers=True, stop_if_static=2):
        torch.export.export(
            model,
            (),
            kwargs=dummy_inputs,
            dynamic_shapes=use_dyn_not_str(dynamic_shapes),
        )
