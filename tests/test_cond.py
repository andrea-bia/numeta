import numpy as np
import numeta as nm


def test_cond():
    @nm.jit
    def cond(a) -> None:
        a[:] = 0.0
        for i in nm.frange(n):
            if nm.cond(i < 9):
                if nm.cond(i < 3):
                    if nm.cond(i < 1):
                        a[i] = 0
                    elif nm.cond(i < 2):
                        a[i] = 1
                    else:
                        a[i] = 2
                    nm.endif()
                elif nm.cond(i < 6):
                    if nm.cond(i < 4):
                        a[i] = 3
                    elif nm.cond(i < 5):
                        a[i] = 4
                    else:
                        a[i] = 5
                    nm.endif()
                else:
                    if nm.cond(i < 7):
                        a[i] = 6
                    elif nm.cond(i < 8):
                        a[i] = 7
                    else:
                        a[i] = 8
                    nm.endif()
                nm.endif()
            elif nm.cond(i < 18):
                if nm.cond(i < 12):
                    if nm.cond(i < 10):
                        a[i] = 9
                    elif nm.cond(i < 11):
                        a[i] = 10
                    else:
                        a[i] = 11
                    nm.endif()
                elif nm.cond(i < 15):
                    if nm.cond(i < 13):
                        a[i] = 12
                    elif nm.cond(i < 14):
                        a[i] = 13
                    else:
                        a[i] = 14
                    nm.endif()
                else:
                    if nm.cond(i < 16):
                        a[i] = 15
                    elif nm.cond(i < 17):
                        a[i] = 16
                    else:
                        a[i] = 17
                    nm.endif()
                nm.endif()
            else:
                if nm.cond(i < 21):
                    if nm.cond(i < 19):
                        a[i] = 18
                    elif nm.cond(i < 20):
                        a[i] = 19
                    else:
                        a[i] = 20
                    nm.endif()
                elif nm.cond(i < 24):
                    if nm.cond(i < 22):
                        a[i] = 21
                    elif nm.cond(i < 23):
                        a[i] = 22
                    else:
                        a[i] = 23
                    nm.endif()
                else:
                    if nm.cond(i < 25):
                        a[i] = 24
                    elif nm.cond(i < 26):
                        a[i] = 25
                    else:
                        a[i] = 26
                    nm.endif()
                nm.endif()
            nm.endif()

    n = 27
    a = np.empty(n, dtype=np.float64)

    cond(a)
    np.testing.assert_allclose(a, np.array(range(n), dtype=np.float64))
