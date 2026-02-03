import numeta as nm
import numpy as np


@nm.jit
def mul(a, b, c, *args):
    for i in nm.range(a.shape[0]):
        for k in nm.range(b.shape[0]):
            c[i, :] += a[i, k] * b[k, :]


a = np.random.rand(100, 100).astype(np.float64)
b = np.random.rand(100, 100).astype(np.float64)
c = np.zeros((100, 100), dtype=np.float64)

mul(a, b, c, 1.0, 1.0)

from numeta.fortran.fortran_syntax import render_stmt_lines

subroutine = mul.get_symbolic_functions()[0]
print("".join(render_stmt_lines(subroutine.get_declaration(), indent=0)))

print(nm.int32[:])
