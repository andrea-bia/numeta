import numpy as np
import numeta as nm


def test_cond():
    @nm.jit
    def cond(a) -> None:
        a[:] = 0.0
        for i in nm.frange(n):
            with nm.If(i < 9):
                with nm.If(i < 3):
                    with nm.If(i < 1):
                        a[i] = 0
                    with nm.ElseIf(i < 2):
                        a[i] = 1
                    with nm.Else():
                        a[i] = 2
                with nm.ElseIf(i < 6):
                    with nm.If(i < 4):
                        a[i] = 3
                    with nm.ElseIf(i < 5):
                        a[i] = 4
                    with nm.Else():
                        a[i] = 5
                with nm.Else():
                    with nm.If(i < 7):
                        a[i] = 6
                    with nm.ElseIf(i < 8):
                        a[i] = 7
                    with nm.Else():
                        a[i] = 8
            with nm.ElseIf(i < 18):
                with nm.If(i < 12):
                    with nm.If(i < 10):
                        a[i] = 9
                    with nm.ElseIf(i < 11):
                        a[i] = 10
                    with nm.Else():
                        a[i] = 11
                with nm.ElseIf(i < 15):
                    with nm.If(i < 13):
                        a[i] = 12
                    with nm.ElseIf(i < 14):
                        a[i] = 13
                    with nm.Else():
                        a[i] = 14
                with nm.Else():
                    with nm.If(i < 16):
                        a[i] = 15
                    with nm.ElseIf(i < 17):
                        a[i] = 16
                    with nm.Else():
                        a[i] = 17
            with nm.Else():
                with nm.If(i < 21):
                    with nm.If(i < 19):
                        a[i] = 18
                    with nm.ElseIf(i < 20):
                        a[i] = 19
                    with nm.Else():
                        a[i] = 20
                with nm.ElseIf(i < 24):
                    with nm.If(i < 22):
                        a[i] = 21
                    with nm.ElseIf(i < 23):
                        a[i] = 22
                    with nm.Else():
                        a[i] = 23
                with nm.Else():
                    with nm.If(i < 25):
                        a[i] = 24
                    with nm.ElseIf(i < 26):
                        a[i] = 25
                    with nm.Else():
                        a[i] = 26

    n = 27
    a = np.empty(n, dtype=np.float64)

    cond(a)
    np.testing.assert_allclose(a, np.array(range(n), dtype=np.float64))
