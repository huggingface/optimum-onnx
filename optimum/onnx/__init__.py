#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# ruff: noqa: F401

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "graph_transformations": [
        "cast_slice_nodes_inputs_to_int32",
        "merge_decoders",
        "remove_duplicate_weights",
        "replace_atenops_to_gather",
        "remove_duplicate_weights_from_tied_info",
    ],
}

if TYPE_CHECKING:
    from .graph_transformations import (
        cast_slice_nodes_inputs_to_int32,
        merge_decoders,
        remove_duplicate_weights,
        remove_duplicate_weights_from_tied_info,
        replace_atenops_to_gather,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
