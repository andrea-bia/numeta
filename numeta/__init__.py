from .types_hint import (
    dtype,
    int32,
    int64,
    float32,
    float64,
    complex64,
    complex128,
    bool8,
    char,
    comptime,
)

integer4 = int32
integer8 = int64
i4 = int32
i8 = int64

real4 = float32
real8 = float64
f4 = float32
f8 = float64
r4 = float32
r8 = float64

complex8 = complex64
complex16 = complex128
c8 = complex64
c16 = complex128

logical1 = bool8
b1 = bool8

from .external_modules import iso_c, omp

from .jit import jit
from .wrappers import *
from .syntax import *
from .settings import settings
