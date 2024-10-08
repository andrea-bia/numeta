# Numeta

Numeta is a simple Just-In-Time (JIT) compiler for Python, focused on metaprogramming. It works like [Numba](https://github.com/numba/numba), but it's much simpler, and the focus is metaprogramming . Although numeta is still in the alpha stage and a personal side project developed in my free time, it aims to create a powerful metaprogramming tool for generating code. For example, I encountered a similar challenge when working on integral evaluations over Gaussian basis functions in computational chemistry.

The focus has been on simplicity, avoiding complex parsing of code, AST, or bytecode, and instead using type hints to differentiate between compiled variables and compile-time variables.

Currently, the code generates Fortran code that is compiled and executed. The obvious reason is that [real programmers want to write FORTRAN programs in any language](https://en.wikipedia.org/wiki/Real_Programmers_Don%27t_Use_Pascal). Additional reasons are discussed in the [Why Fortran Backend](#why-fortran-backend) section.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Type Hints](#type-hints)
  - [Compile-Time Variables](#compile-time-variables)
  - [Parallelizing Loops](#parallelizing-loops)
- [Examples](#examples)
  - [First For Loop](#first-for-loop)
  - [Conditional Statements](#conditional-statements)
  - [How to Link an External Library](#how-to-link-an-external-library)
  - [Parallel Loop Example](#parallel-loop-example)
- [Why Fortran Backend?](#why-fortran-backend)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Metaprogramming Focus**: Leverages metaprogramming for flexible code generation.
- **Simplified Approach**: Does not rely on parsing AST or bytecode, making it straightforward.
- **Type Annotations**: Uses Python's type hints to differentiate between compiled and compile-time variables.

## Installation

To install numeta, use:

```bash
git clone https://gitlab.com/andrea_bianchi/numeta
cd numeta
pip install .
```

## Quick Start

Here's a quick example demonstrating how numeta works:

```python
import numeta as nm

@nm.jit
def mixed_loops(n, array: nm.f8[:, :]) -> None:
    for i in range(n):
        for j in nm.frange(n):
            array[j, i] = i + j
```

This code runs as usual Python code. The first loop (`n`) is a compile-time variable and will be unrolled, while the second loop will be compiled and executed as Fortran code. The generated Fortran code looks like this:

```fortran
subroutine mixed_loops(array) bind(C)
    use iso_c_binding, only: c_double
    use iso_c_binding, only: c_int64_t
    implicit none
    real(c_double), dimension(0:9, 0:9), intent(inout) :: array
    integer(c_int64_t) :: fc_i1
    integer(c_int64_t) :: fc_i2
    integer(c_int64_t) :: fc_i3
    do fc_i1 = 0_c_int64_t, 2_c_int64_t
        array(0, fc_i1) = (0_c_int64_t + fc_i1)
    end do
    do fc_i2 = 0_c_int64_t, 2_c_int64_t
        array(1, fc_i2) = (1_c_int64_t + fc_i2)
    end do
    do fc_i3 = 0_c_int64_t, 2_c_int64_t
        array(2, fc_i3) = (2_c_int64_t + fc_i3)
    end do
end subroutine mixed_loops
```

This is where one can appreciate the beauty of Fortran.
Note that the indices are reversed because Fortran arrays are column-major, meaning the first index is the column and the second is the row.

## Usage

### Type Hints

In numeta, to differentiate compile-time variables and runtime variables, you should use type hints. This allows for a clear separation between the two and enables metaprogramming capabilities.

Currently, the following types are supported:

- `int32`, `int64` (also aliased as `integer4`, `integer8`, `i4`, `i8`)
- `float32`, `float64` (also aliased as `real4`, `real8`, `f4`, `f8`, `r4`, `r8`)
- `complex64`, `complex128` (also aliased as `complex8`, `complex16`, `c8`, `c16`)
- `bool8` (also aliased as `logical1`, `b1`)
- `char`
- `dtype` (for structured numpy arrays)

#### How to Use Type Hints

Examples of using type hints in numeta:

- `nm.f8[:, :]` defines a rank-2 array of `float64` numbers.
- `nm.f8[20, 20]` defines a fixed shape array of `float64` with dimensions 20x20.
- To specify the memory layout, use `nm.f8[:, :]['F']` or `nm.f8[:, :]['C']` for Fortran or C order, respectively.
- You can also convert from numpy types, e.g., `nm.dtype[np.float64][:]`.
- To create structured arrays, convert from numpy `dtype` or define using a tuple, e.g., `nm.dtype[('name1', dtype1, dim1), ('name2', dtype2)]` or with a dictionary.

### Compile-Time Variables

Variables without type annotations are considered compile-time constants. They are evaluated during the compilation phase and can be used to control code generation. Compile-time variables should be hashable; therefore, you can't use a list or a dictionary directly. If needed, map them to a hashable variable and call it in the code.

Incorrect usage:

```python
to_sum_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
to_sum_2 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

@nm.jit
def compile_time_variable(to_sum, array: nm.f8[10]) -> None:
    for i in range(10):
        array[i] += to_sum
```

Correct usage:

```python
to_sum = {}
to_sum[1] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
to_sum[2] = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

@nm.jit
def compile_time_variable(idx_sum, array: nm.f8[10]) -> None:
    for i in range(10):
        array[i] += to_sum[idx_sum][i]
```

## Examples

### First For Loop

Below is a simple example of a for loop using numeta:

```python
import numeta as nm

@nm.jit
def first_for_loop(n: nm.i8, array: nm.f8[:]) -> None:
    for i in nm.frange(n):
        array[i] = i * 2
```

In this example:

- `n` is the size of the array.
- `array` is a rank-1 array of `float64` numbers.
- The loop runs using `nm.frange(n)` to generate a compiled loop that performs the operation.

Alternatively, you can use:

```python
@nm.jit
def do_loop(n: nm.i8, array: nm.f8[:]) -> None:
    i = nm.scalar(nm.i8)
    with nm.do(i, 0, n - 1):
        array[i] = i * 2
```

This approach uses `nm.do` to emulate the Fortran `do` loop style.

### Conditional Statements

Below is an example of how to use conditional statements with numeta:

```python
import numeta as nm

@nm.jit
def conditional_example(n: nm.i8, array: nm.f8[:]) -> None:
    for i in nm.frange(n):
        if nm.cond(i < 1):
            array[i] = 0
        elif nm.cond(i < 2):
            array[i] = 1
        else:
            array[i] = 2
        nm.endif()
```

Note: You need to use `nm.endif()` at the end of the conditional block, though I'm working on improving this syntax to make it more intuitive. 
It is currently difficult to maintain Python-like syntax for generated conditional code because some branches may never be taken, which complicates the code generation and obligates to read the AST.

Alternatively, you can use:

```python
with nm.If(i < 3):
    with nm.If(i < 1):
        array[i] = 0
    with nm.ElseIf(i < 2):
        array[i] = 1
    with nm.Else():
        array[i] = 2
```

This approach is safer, albeit less elegant, and will generate the same code as the previous example without needing to read the AST.

### How to Link an External Library

Below is an example of how to link an external library, specifically linking LAPACK (very alpha):

```python
import numeta as nm

# Create an external library wrapper for LAPACK
lapack = nm.ExternalLibraryWrapper("lapack")

# Add a method from LAPACK to the wrapper
lapack.add_method(
    "dgemm",
    [
        nm.char,      # transa
        nm.char,      # transb
        nm.i4,        # m
        nm.i4,        # n
        nm.i4,        # k
        nm.f8,        # alpha
        nm.f8[:],     # a
        nm.i4,        # lda
        nm.f8[:],     # b
        nm.i4,        # ldb
        nm.f8,        # beta
        nm.f8[:],     # c
        nm.i4         # ldc
    ],
    None,
    bind_c=False
)

# Set up a compile-time constant
n = 100

# Configure numeta settings
nm.settings.set_integer(32)

@nm.jit
def matmul(a: nm.f8[:, :], b: nm.f8[:, :], c: nm.f8[:, :]):
    # Call the linked LAPACK dgemm method
    lapack.dgemm("N", "N", n, n, n, 1.0, b, n, a, n, 0.0, c, n)
```

In this example:

- `lapack = nm.ExternalLibraryWrapper("lapack")` creates a wrapper for the LAPACK library.
- `lapack.add_method()` adds the `dgemm` method for matrix multiplication.
- The method signature includes parameters such as matrix dimensions and scalars.
- The `matmul` function then uses `lapack.dgemm` to perform matrix multiplication.

### Parallel Loop Example

numeta provides the capability to parallelize loops using `nm.prange`. This closely follows the OpenMP `parallel do` model, allowing for efficient parallel execution by leveraging shared and private variables. Note that this feature is still in alpha, and the syntax might change in the future.

Below is an example of how to parallelize a loop using `nm.prange`:

```python
import numeta as nm

@nm.jit
def pmul(a: nm.dtype[dtype][:, :], b: nm.dtype[dtype][:, :], c: nm.dtype[dtype][:, :]):
    for i in nm.prange(a.shape[0], default='private', shared=[a, b, c], schedule='static'):
        for k in nm.frange(b.shape[0]):
            c[i, :] += a[i, k] * b[k, :]
```

In this example:

- `nm.prange(a.shape[0])` parallelizes the outer loop.
- `default='private'` specifies that loop variables are private by default.
- The `shared` list includes variables that are shared across threads.
- The `schedule='static'` controls the scheduling of loop iterations.

## Why Fortran Backend?

I chose to use Fortran as the backend for numeta because:

1. **Familiarity**: I have experience with Fortran, which made it easier to implement.
2. **Native Array Operations**: Fortran supports array operations natively, reducing the amount of code required to support them.
3. **Fast Compilation**: Fortran is relatively fast to compile, which is beneficial for JIT compilation.

While Fortran has some limitations, it allowed me to create a working prototype quickly. I'm open to exploring other backends in the future and improving the generated code, so suggestions are welcome.

## Contributing

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
