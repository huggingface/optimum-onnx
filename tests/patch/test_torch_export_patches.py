import unittest
from typing import Any

import numpy as np

from optimum.torch_export_patches._core import torch_export_patches


class TestOnnxExportErrors(unittest.TestCase):
    def assertEqualArrayAny(self, expected: Any, value: Any, atol: float = 0, rtol: float = 0, msg: str = ""):  # noqa: N802
        if isinstance(expected, (tuple, list, dict)):
            self.assertIsInstance(value, type(expected), msg=msg)
            self.assertEqual(len(expected), len(value), msg=msg)
            if isinstance(expected, dict):
                for k in expected:
                    self.assertIn(k, value, msg=msg)
                    self.assertEqualArrayAny(expected[k], value[k], msg=msg, atol=atol, rtol=rtol)
            else:
                excs = []
                for i, (e, g) in enumerate(zip(expected, value)):
                    try:
                        self.assertEqualArrayAny(e, g, msg=msg, atol=atol, rtol=rtol)
                    except AssertionError as e:
                        excs.append(f"Error at position {i} due to {e}")
                if excs:
                    msg_ = "\n".join(excs)
                    msg = f"{msg}\n{msg_}" if msg else msg_
                    raise AssertionError(f"Found {len(excs)} discrepancies\n{msg}")
        elif expected.__class__.__name__ in ("DynamicCache", "StaticCache"):
            atts = {"key_cache", "value_cache"}
            self.assertEqualArrayAny(
                {k: expected.__dict__.get(k, None) for k in atts},
                {k: value.__dict__.get(k, None) for k in atts},
                atol=atol,
                rtol=rtol,
            )
        elif isinstance(expected, (int, float, str)):
            self.assertEqual(expected, value, msg=msg)
        elif hasattr(expected, "shape"):
            self.assertEqual(type(expected), type(value), msg=msg)
            self.assertEqualArray(expected, value, msg=msg, atol=atol, rtol=rtol)
        elif expected is None:
            assert value is None, f"Expected is None but value is of type {type(value)}"
        else:
            raise AssertionError(f"Comparison not implemented for types {type(expected)} and {type(value)}")

    def assertEqualArray(  # noqa: N802
        self,
        expected: Any,
        value: Any,
        atol: float = 0,
        rtol: float = 0,
        msg: str | None = None,
    ):
        if hasattr(expected, "detach") and hasattr(value, "detach"):
            if msg:
                try:
                    self.assertEqual(expected.dtype, value.dtype)
                except AssertionError as e:
                    raise AssertionError(msg) from e
                try:
                    self.assertEqual(expected.shape, value.shape)
                except AssertionError as e:
                    raise AssertionError(msg) from e
            else:
                self.assertEqual(expected.dtype, value.dtype)
                self.assertEqual(expected.shape, value.shape)

            import torch

            try:
                torch.testing.assert_close(value, expected, atol=atol, rtol=rtol)
            except AssertionError as e:
                expected_max = torch.abs(expected).max()
                expected_value = torch.abs(value).max()
                rows = [
                    f"{msg}\n{e}" if msg else str(e),
                    f"expected max value={expected_max}",
                    f"expected computed value={expected_value}",
                ]
                raise AssertionError("\n".join(rows))
            return

        if hasattr(expected, "detach"):
            expected = expected.detach().cpu().numpy()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if msg:
            try:
                self.assertEqual(expected.dtype, value.dtype)
            except AssertionError as e:
                raise AssertionError(msg) from e
            try:
                self.assertEqual(expected.shape, value.shape)
            except AssertionError as e:
                raise AssertionError(msg) from e
        else:
            self.assertEqual(expected.dtype, value.dtype)
            self.assertEqual(expected.shape, value.shape)

        try:
            np.testing.assert_allclose(desired=expected, actual=value, atol=atol, rtol=rtol)
        except AssertionError as e:
            expected_max = np.abs(expected).max()
            expected_value = np.abs(value).max()
            te = expected.astype(int) if expected.dtype == np.bool_ else expected
            tv = value.astype(int) if value.dtype == np.bool_ else value
            rows = [
                f"{msg}\n{e}" if msg else str(e),
                f"expected max value={expected_max}",
                f"expected computed value={expected_value}\n",
                f"ratio={te / tv}\ndiff={te - tv}",
            ]
            raise AssertionError("\n".join(rows))

    # @skipif_ci_windows("not working on Windows")
    def test_pytree_flatten_mamba_cache(self):
        import torch
        import torch.utils._pytree as py_pytree

        try:
            from transformers.models.mamba.modeling_mamba import MambaCache
        except ImportError:
            from transformers.cache_utils import MambaCache

        class Config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        cache = MambaCache(Config(), max_batch_size=1, device="cpu")

        with torch_export_patches(verbose=1):
            values, spec = py_pytree.tree_flatten(cache)
            cache2 = py_pytree.tree_unflatten(values, spec)
            self.assertEqual(cache.max_batch_size, cache2.max_batch_size)
            self.assertEqual(cache.intermediate_size, cache2.intermediate_size)
            self.assertEqual(cache.ssm_state_size, cache2.ssm_state_size)
            self.assertEqual(cache.conv_kernel_size, cache2.conv_kernel_size)
            self.assertEqualArrayAny(cache.conv_states, cache2.conv_states)
            self.assertEqualArrayAny(cache.ssm_states, cache2.ssm_states)

    # @skipif_ci_windows("not working on Windows")
    def test_exportable_mamba_cache(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class Config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 64
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(Config(), max_batch_size=1, device="cpu")
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)

        with torch_export_patches(verbose=1, patch_transformers=True):
            cache = MambaCache(Config(), max_batch_size=1, device="cpu")
            torch.export.export(Model(), (x, cache))

    # @skipif_ci_windows("not working on Windows")
    def test_exportable_mamba_cache_dynamic(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaCache

        class Config:
            def __init__(self):
                self.intermediate_size = 8
                self.state_size = 16
                self.conv_kernel = 32
                self.num_hidden_layers = 2
                self.dtype = torch.float16

        class Model(torch.nn.Module):
            def forward(self, x: torch.Tensor, cache: MambaCache):
                x1 = cache.ssm_states[0] + x
                x2 = cache.conv_states[0][:, :, ::2] + x1
                return x2

        cache = MambaCache(Config(), max_batch_size=1, device="cpu")
        x = torch.ones(2, 8, 16).to(torch.float16)
        model = Model()
        model(x, cache)
        dynamic = torch.export.Dim.DYNAMIC

        with torch_export_patches():
            cache = MambaCache(Config(), max_batch_size=2, device="cpu")
            torch.export.export(
                Model(),
                (x, cache),
                dynamic_shapes=({0: dynamic}, [[{0: dynamic}, {0: dynamic}], [{0: dynamic}, {0: dynamic}]]),
            )

    def test_exportable_dynamic_shapes_constraints(self):
        import torch

        class CustomCache:
            def __init__(self, shape=None):
                self.cache = [torch.zeros(shape), torch.zeros(shape)] if shape else []

        def flatten_cache(cache):
            return [cache.cache], ["cache"]

        def unflatten_cache(values, context, output_type=None):
            cache = CustomCache()
            cache.cache = values[0]
            return cache

        def flatten_with_keys_cache(d):
            values, context = flatten_cache(d)
            return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context

        torch.utils._pytree.register_pytree_node(
            CustomCache,
            flatten_cache,
            unflatten_cache,
            serialized_type_name=f"{CustomCache.__module__}.{CustomCache.__name__}",
            flatten_with_keys_fn=flatten_with_keys_cache,
        )

        class Model(torch.nn.Module):
            def forward(self, x, cache):
                return cache.cache[0][0, :] + x

        model = Model()
        model.eval()
        x, cache = torch.rand((2, 4)), CustomCache((2, 4))
        model(x, cache)
        dynamic = torch.export.Dim.DYNAMIC
        torch.export.export(model, (x, cache), dynamic_shapes=({0: dynamic}, [[{0: dynamic}, {0: dynamic}]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
