# TODO

## feat

## refactor
- Do not return the array shape in `return` statements if the shape is known at compile time.
- Improve efficiency of nested function returns by avoiding unnecessary allocations. 

## fix
- Fix name conflicts when inlining a function that creates temporary variables and the caller also creates temporary variables. This happens because `BuilderHelper` starts counting from zero.
  Possible solutions:
    - Make temporary variable names start with the function name (ugly) or a counter. (easy)
    - Instead of creating a symbolic function and inlining it, call the function body directly. (hard, overhead, difficult to estimate inlining threshold)
