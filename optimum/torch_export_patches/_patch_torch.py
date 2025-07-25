import inspect
import os
from typing import Any, Callable
import torch


def catch_produce_guards_and_solve_constraints(
    previous_function: Callable,
    fake_mode: "FakeTensorMode",
    gm: "torch.fx.GraphModule",
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    equalities_inputs: "EqualityConstraint",
    original_signature: inspect.Signature,
    _is_torch_jit_trace: bool = False,
    verbose: int = 0,
):
    try:
        return previous_function(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_catch_produce_guards_and_solve_constraints] ERROR: "
                f"produce_guards_and_solve_constraints failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"fake_mode={fake_mode}\n"
                f"dynamic_shapes={dynamic_shapes}\n"
                f"equalities_inputs={equalities_inputs}\n"
                f"original_signature={original_signature}\n"
                f"_is_torch_jit_trace={_is_torch_jit_trace}\n"
                f"exc={e}\ngm={gm}"
            )
        torch._dynamo.reset()


def patch__check_input_constraints_for_graph(
    previous_function: Callable,
    input_placeholders: list[torch.fx.Node],
    flat_args_with_path,
    range_constraints,
    verbose: int = 0,
) -> None:
    try:
        return previous_function(input_placeholders, flat_args_with_path, range_constraints)
    except Exception as e:
        if not int(os.environ.get("SKIP_SOLVE_CONSTRAINTS", "1")):
            raise
        if verbose:
            print(
                f"[_check_input_constraints_for_graph] ERROR: "
                f"_check_input_constraints_for_graph failed, "
                f"use SKIP_SOLVE_CONSTRAINTS=0 to avoid skipping\n"
                f"input_placeholders={input_placeholders}\n"
                f"range_constraints={range_constraints}\n"
                f"exc={e}"
            )
        torch._dynamo.reset()


def patched_infer_size(a, b):
    """Patches ``torch._subclasses.fake_impls.infer_size``."""
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    dimsa = len(a)
    dimsb = len(b)
    ndim = max(dimsa, dimsb)
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dima = dimsa - 1 - offset
        dimb = dimsb - 1 - offset
        sizeA = a[dima] if dima >= 0 else 1
        sizeB = b[dimb] if dimb >= 0 else 1

        # NB: It is very important to test for broadcasting, before testing
        # sizeA == sizeB.  This is because the broadcasting tests are likely
        # to be statically known (in particular, if sizeA/sizeB is unbacked
        # but size-like, we will unsoundly assume they never equal 1), but
        # the sizeA == sizeB test may not be statically known.  However, once
        # we have established that no broadcasting is happening, the
        # sizeA == sizeB is now expect_true and we can defer it as a runtime
        # assert (this works because Python will return the terminal
        # expression of an or statement as-is, without bool()'ing it; if this
        # were not the case, we'd need to write this using torch.sym_or() or
        # something like that).
        try:
            b1 = guard_size_oblivious(sizeA == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b1 = False
        try:
            b2 = guard_size_oblivious(sizeB == 1)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b2 = False
        try:
            b3 = guard_size_oblivious(sizeA == sizeB)
        except torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
            b3 = False
        if b1 or b2 or b3:
            expandedSizes[i] = sizeB if guard_size_oblivious(sizeA == 1) else sizeA
        else:
            # In this case, the current implementation of torch fails (17/12/2024).
            # Try model SmolLM.
            expandedSizes[i] = torch.sym_max(sizeA, sizeB)
    return tuple(expandedSizes)


def patched__broadcast_shapes(*_shapes):
    """Patches ``torch._refs._broadcast_shapes``."""
    from functools import reduce
    from torch._prims_common import IntLike
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    shapes = tuple(
        (x,) if isinstance(x, IntLike) else x for x in filter(lambda x: x is not None, _shapes)
    )

    # Short-circuits on no input
    if len(shapes) == 0:
        return None

    # Type checking
    # TODO: make common validations available as utils
    for shape in shapes:
        assert isinstance(shape, Sequence)

    # Computes common shape
    common_shape = [  # List[Union[int, torch.SymInt]]
        1,
    ] * reduce(max, (len(shape) for shape in shapes))
    for _arg_idx, shape in enumerate(shapes):
        for idx in range(-1, -1 - len(shape), -1):
            if guard_size_oblivious(common_shape[idx] == 1):
                if shape[idx] < 0:
                    raise ValueError(
                        "Attempting to broadcast a dimension with negative length!"
                    )
                common_shape[idx] = shape[idx]
            elif guard_size_oblivious(shape[idx] != 1):
                common_shape[idx] = torch.sym_max(common_shape[idx], shape[idx])

    return common_shape


class patched_ShapeEnv:

    def _check_frozen(
        self, expr: "sympy.Basic", concrete_val: "sympy.Basic"  # noqa: F821
    ) -> None:
        if self.frozen:
            self.counter["ignored_backward_guard"] += 1
            raise AssertionError(
                f"[patched_ShapeEnv] Ignored guard {expr} == {concrete_val}, "
                f"this could result in accuracy problems."
            )

    def _set_replacement(
        self, a: "sympy.Symbol", tgt: "sympy.Expr", msg: str  # noqa: F821
    ) -> None:
        """
        Adds or updates a replacement for a symbol.
        Use this instead of `self.replacements[a] = tgt`.
        """
        if tgt == self.replacements.get(a, None):
            return

        if a in tgt.free_symbols:
            return

        import sympy
        from torch._logging import structured
        from torch.utils._traceback import CapturedTraceback
        from torch._logging import trace_structured
        from torch._guards import TracingContext
        from torch.utils._sympy.functions import FloorToInt, CeilToInt
        from torch.utils._sympy.solve import try_solve
        from torch.fx.experimental.symbolic_shapes import (
            _is_supported_equivalence,
            ValueRanges,
        )

        # Precondition: a == tgt
        assert isinstance(a, sympy.Symbol)

        if self.allow_complex_guards_as_runtime_asserts and not _is_supported_equivalence(tgt):
            # continuing leads to placeholder shapes
            # having complex expressions that we can't resolve
            return

        # Handles nested tensor symbolic variables which don't have
        # var_to_range bounds
        tgt_bound = None
        if a in self.var_to_range:
            src_bound = self.var_to_range[a]

            # First, refine the value range of a based on the computed value range
            # of tgt.  This is always OK to do, even if we decide not to do the
            # substitution in the end.  This might be a no-op, if a already has
            # a tighter bound
            tgt_bound = self.bound_sympy(tgt)
            self._update_var_to_range(a, tgt_bound)

            # Next, check if we can update the range of free symbols in tgt
            # based on the range in a. But only do it if:
            #  - the source bound non-trivially improves over what we get out of
            #    the existing bounds.
            #  - the replacement is univariate and we can invert the tgt expression
            if not tgt_bound.issubset(src_bound) and len(tgt.free_symbols) == 1:
                b = next(iter(tgt.free_symbols))
                # Try to invert the equality
                r = try_solve(sympy.Eq(a, tgt), b, floordiv_inequality=False)
                if r is not None:
                    self.log.debug(
                        "set_replacement: solve for %s in %s == %s gives %s",
                        b,
                        a,
                        tgt,
                        r,
                    )
                    # The solution here can be non-integral, for example, if
                    # we have s0 = 2*s1, then s1 = s0/2.  What we would like
                    # to do is calculated the bounds in arbitrary precision,
                    # and then requantize the bound to integers when we are
                    # done.
                    rat_b_bound = self.bound_sympy(r[1])
                    b_bound = ValueRanges(
                        CeilToInt(rat_b_bound.lower), FloorToInt(rat_b_bound.upper)
                    )
                    self._update_var_to_range(b, b_bound, self.var_to_range_sloc[a])
                    tgt_bound = self.bound_sympy(tgt)
                    assert tgt_bound.issubset(
                        src_bound
                    ), f"{tgt_bound=} not a subset of {src_bound=}"

            # TODO: Should we propagate size-like-ness?
            #
            # Pros: if u0 is size-like, intuitively u0 == u1 should cause u1
            # to become size-like.
            #
            # Cons: if u0 is size-like, what about u0 - 1 == u1?  You CAN'T
            # propagate in this case, because what if u0 == 0, then u1 is negative
            # and clearly isn't a size.  So, at minimum, any f(x) whose value
            # range isn't [0, inf] given x in [0, inf] cannot propagate
            # size-like-ness.  But there are many situations where you could
            # imagine u1 is going to be size-like and actually you just didn't
            # have a refined enough value range on u0.  Since even innocuous
            # looking arithmetic operations can destroy size-like-ness, it's
            # best to not propagate it at all and force the user to annotate it
            # as necessary.
            #
            # Compromise: we preserve size-like-ness only for exact equality
            # and nothing else.
            if a in self.size_like and isinstance(tgt, sympy.Symbol):
                self.size_like.add(tgt)
            elif isinstance(tgt, sympy.Symbol) and tgt in self.size_like:
                self.size_like.add(a)

            # Now, decide if we will do the substitution.
            #
            #  - If the source has a non-trivial range, only substitute if
            #    we preserve this range.  Note that we may have propagated
            #    the src_range to free variables in tgt when tgt is univariate
            #    and we could find an inverse, which helps us achieve this.
            #    This ensures we never "forget" about user defined ranges,
            #    even if they end up being defined on composite formulas
            #    like s0 + s1.
            #
            #  - If the variable is unbacked, only substitute if the substitution
            #    would preserve the bounds also under size-like-ness conditions.

            if not tgt_bound.issubset(src_bound):
                self.log.debug(
                    "skipped set_replacement %s = %s (%s) [%s not subset of %s]",
                    a,
                    tgt,
                    msg,
                    tgt_bound,
                    src_bound,
                )
                return
            elif a in self.size_like:
                tgt_bound_so = self.bound_sympy(tgt, size_oblivious=True)
                src_bound_so = self.bound_sympy(a, size_oblivious=True)
                if not tgt_bound_so.issubset(src_bound_so):
                    self.log.debug(
                        "skipped set_replacement %s = %s (%s) "
                        "[%s not subset of %s (size-oblivious conditions)]",
                        a,
                        tgt,
                        msg,
                        tgt_bound_so,
                        src_bound_so,
                    )
                    return

        if isinstance(tgt, (sympy.Integer, sympy.Float)):
            # specializing to a constant, which is likely unexpected (unless
            # you specified dynamic=True)

            user_tb = TracingContext.extract_stack()
            trace_structured(
                "symbolic_shape_specialization",
                metadata_fn=lambda: {
                    "symbol": repr(a),
                    "sources": [s.name() for s in self.var_to_sources.get(a, [])],
                    "value": repr(tgt),
                    "reason": msg,
                    "stack": structured.from_traceback(
                        CapturedTraceback.extract(skip=1).summary()
                    ),
                    "user_stack": (structured.from_traceback(user_tb) if user_tb else None),
                },
            )

            # if config.print_specializations:
            #    self.log.warning(
            #         "Specializing %s to %s", self.var_to_sources[a][0].name(), tgt
            #     )
            #     self.log.debug("SPECIALIZATION", stack_info=True)
        assert msg != "range_refined_to_singleton", (
            f"patched_ShapeEnv: A dynamic dimension becomes static! "
            f"a={a!r}, tgt={tgt!r}, msg={msg!r}, tgt_bound={tgt_bound}"
        )
        # log.info("set_replacement %s = %s (%s) %s", a, tgt, msg, tgt_bound)
        self.replacements[a] = tgt
        # NB: the replacement may get refined, but the user will find the
        # FIRST one most useful (TODO: Maybe we could consider tracking all of
        # them)
        if a not in self.replacements_slocs:
            self.replacements_slocs[a] = self._get_sloc()
        self._update_version_counter()

        # When specializing 'a == tgt', the equality should be also conveyed to
        # Z3, in case an expression uses 'a'.
        self._add_target_expr(sympy.Eq(a, tgt, evaluate=False))

    def _log_guard(
        self, prefix: str, g: "SympyBoolean", forcing_spec: bool  # noqa: F821
    ) -> None:
        self._log_guard_remember(prefix=prefix, g=g, forcing_spec=forcing_spec)


def patched_vmap(func, in_dims=0, out_dims=0):
    """Python implementation of :func:`torch.vmap`.

    The implementation raises an issue when it is being exported with
    :func:`torch.export.export` when the function is called with
    non tensors arguments and the batch size is dynamic.
    """

    def wrapped(*args):
        in_dims_ = (
            ([in_dims] * len(args))
            if not isinstance(in_dims, (list, tuple))
            else list(in_dims)
        )
        batch_size = None
        batched_args = []
        for arg, in_dim in zip(args, in_dims_):
            if in_dim is None:
                batched_args.append(arg)
                continue

            assert batch_size is None or batch_size == arg.size(in_dim), (
                f"Unable to continue, batch_size={batch_size}, in_dim={in_dim}, "
                f"arg.size(in_dim)={arg.size(in_dim)}"
            )
            if batch_size is None:
                batch_size = arg.size(in_dim)
            arg = arg.movedim(in_dim, 0)
            batched_args.append(arg)

        if all(isinstance(a, torch.Tensor) for a in args) and isinstance(
            batch_size, torch.SymInt
        ):
            batched_tensors = [
                (
                    arg
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else arg.unsqueeze(0).expand((batch_size, *arg.shape))
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]
            results = torch.ops.higher_order.scan(func, [], batched_tensors, [])
            stacked = results[0]
            if out_dims != 0:
                return stacked.movedim(0, out_dims)
            return stacked

        else:
            torch._check(
                not isinstance(batch_size, torch.SymInt),
                lambda: (
                    f"patched_vmap supports dynamic batch_size only if all argument "
                    f"are tensors but types are {[type(a) for a in args]}"
                ),
            )
            batched_tensors = [
                (
                    (None, arg)
                    if (isinstance(arg, torch.Tensor) and in_dim is not None)
                    else (arg, arg)
                )
                for arg, in_dim in zip(batched_args, in_dims_)
            ]

            results = []
            for i in range(batch_size):
                input_slice = [v if v is not None else arg[i] for v, arg in batched_tensors]
                result = func(*input_slice)
                results.append(result)

            if isinstance(results[0], torch.Tensor):
                stacked = torch.stack(results)
                if out_dims != 0:
                    return stacked.movedim(0, out_dims)
                return stacked
            return results

    return wrapped
