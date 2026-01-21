# TODO

## feat
- Add variable support for NumetaLibrary.

## refactor
- Do not return the array shape in `return` statements if the shape is known at compile time.
- Improve efficiency of nested function returns by avoiding unnecessary allocations.

## fix
- When loading a NumetaLibrary must fix name collisions of functions / structs / ? .
- Create interface for functions to be to include function from external libraries that need interfaces.
