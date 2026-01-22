# TODO

## feat
- Add variable support for NumetaLibrary.

## refactor
- Do not return the array shape in `return` statements if the shape is known at compile time.
- Improve efficiency of nested function returns by avoiding unnecessary allocations.

## fix
- Create interfaces so functions can include external-library functions that need interfaces.
- Check collisions between NumetaLibrary and CompiledFunction libraries because they cannot be loaded at the same time (also PyCExtension).
