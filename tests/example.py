import numeta as nm
import numpy as np


@nm.jit(directory="example")
def test(n: nm.CT, a):
    for i in range(n):
        for j in nm.frange(a.shape[0]):
            a[j] += i * j


n = 3
m = 1000
a = np.zeros(m, dtype=np.int32)

test(n, a)
