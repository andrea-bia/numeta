# TODO

## feat
- [ ] Add variable support for `NumetaLibrary`.
- [ ] Cross-platform shared library support: detect `.so`/`.dylib`/`.dll` in `compiler.py`. Make compiler names configurable via settings or env vars (`NUMETA_FC`, `NUMETA_CC`).

## refactor
- [ ] Do not return the array shape in `return` statements if the shape is known at compile time.
- [ ] Improve efficiency of nested function returns by avoiding unnecessary allocations.
- [ ] Create `BaseEmitter` / visitor pattern abstraction to eliminate Fortran/C emitter duplication. Convert `_render_intrinsic` if/elif chain to a dispatch table.
- [ ] Reduce global mutable state: convert `BuilderHelper.current_builder`, `CondHelper.curr`, `_n_global_constant` to `contextvars.ContextVar`.
- [ ] Replace wildcard imports (`from .wrappers import *`, `from .ast import *`) with explicit imports and define `__all__` lists.
- [ ] Extract shared argument-parsing logic from `wrappers/range.py` and `wrappers/prange.py` into a helper.
- [ ] Consolidate `DataType._get_bind_c_type` — move the duplicated classmethod from 10 subclasses into the base class with a lookup table.
- [ ] Move lazy imports to module level in `ast/expressions/expression_node.py` and `c/emitter.py`.

## fix

## Code Comments to Resolve

### TODO
- [ ] `numeta/fortran/external_modules/omp.py:43` - Should we add automatically variables declared inside the loop?
- [ ] `numeta/fortran/external_modules/omp.py:50` - Should we automatically array shapes if explicitly declared as variables?
- [ ] `numeta/ast/expressions/binary_operation_node.py:68` - Too slow (in `__bool__` method)
- [ ] `numeta/ast/expressions/binary_operation_node.py:87` - Too slow (in `__bool__` method)
- [ ] `numeta/ast/namespace.py:103` - Arguments are not used but it could be used to check if the arguments are correct
- [x] `numeta/builder_helper.py:37-42` - Deprecate `generate_local_variables` classmethod, use instance method instead
- [ ] `numeta/wrappers/constant.py:52-54` - parameter is not supported yet, so not really constant
- [ ] `numeta/wrappers/constant.py:63-65` - parameter is not supported yet, so not really constant
- [ ] `numeta/wrappers/declare_global_constant.py:46-47` - parameter is not supported yet, so not really constant
- [ ] `tests/test_syntax.py:162` - TODO(Shape, 1, "shape")
- [ ] `tests/test_syntax.py:591` - TODO(Shape, 1, "shape")

### HACK
- [ ] `numeta/ast/expressions/getitem.py:88` - Fortran does not treat temporary variables as first class citizens (only for ArrayConstructor)
- [ ] `numeta/ast/statements/variable_declaration.py:27-29` - Non standard array lower bound so we have to shift it
- [ ] `numeta/wrappers/cond.py:41-46` - add a try except block to take care of the indentation

## chore
- [ ] Add `[tool.mypy]` configuration to `pyproject.toml` and incrementally reduce `Any` usage.
- [ ] Pin `numpy` version bounds and add `[project.optional-dependencies]` for dev/test tooling.
- [ ] Add module-level and class-level docstrings throughout. Fix misleading docstrings in `compiler.py`.
- [ ] Resolve or triage the 14 TODO/FIXME/HACK comments across the codebase.
- [ ] Add `ruff` to `.pre-commit-config.yaml` alongside Black.
