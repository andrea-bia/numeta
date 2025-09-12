import numpy as np
import pytest
import numeta as nm


def test_optional_argument():
    @nm.jit
    def fill(a, value=1.0):
        a[:] = value

    a = np.empty((5,), dtype=np.float64)
    fill(a)
    np.testing.assert_allclose(a, np.full((5,), 1.0, dtype=np.float64))
    fill(a, 2.0)
    np.testing.assert_allclose(a, np.full((5,), 2.0, dtype=np.float64))
    fill(a, value=3.0)
    np.testing.assert_allclose(a, np.full((5,), 3.0, dtype=np.float64))

    assert len(fill.get_symbolic_functions()) == 1


def test_optional_comptime_argument():
    @nm.jit
    def fill(a, value: nm.comptime = 1.0):
        a[:] = value

    a = np.empty((5,), dtype=np.float64)
    fill(a)
    np.testing.assert_allclose(a, np.full((5,), 1.0, dtype=np.float64))
    fill(a, 2.0)
    np.testing.assert_allclose(a, np.full((5,), 2.0, dtype=np.float64))
    fill(a, value=3.0)
    np.testing.assert_allclose(a, np.full((5,), 3.0, dtype=np.float64))

    assert len(fill.get_symbolic_functions()) == 3


def test_optional_argument_mixed():
    @nm.jit
    def fill(a, value=1.0, value2=-2.0):
        a[:] = value
        a[2] = value2

    a = np.empty((5,), dtype=np.float64)
    fill(a)
    expected = np.full((5,), 1.0, dtype=np.float64)
    expected[2] = -2.0
    np.testing.assert_allclose(a, expected)

    fill(a, 2.0)
    expected = np.full((5,), 2.0, dtype=np.float64)
    expected[2] = -2.0
    np.testing.assert_allclose(a, expected)

    fill(a, value=3.0)
    expected = np.full((5,), 3.0, dtype=np.float64)
    expected[2] = -2.0
    np.testing.assert_allclose(a, expected)

    fill(a, value2=2.0)
    expected = np.full((5,), 1.0, dtype=np.float64)
    expected[2] = 2.0
    np.testing.assert_allclose(a, expected)

    fill(a, value2=2.0, value=5.0)
    expected = np.full((5,), 5.0, dtype=np.float64)
    expected[2] = 2.0
    np.testing.assert_allclose(a, expected)

    fill(a, 2.0, 5.0)
    expected = np.full((5,), 2.0, dtype=np.float64)
    expected[2] = 5.0
    np.testing.assert_allclose(a, expected)

    assert len(fill.get_symbolic_functions()) == 1


@pytest.mark.parametrize("n_args", range(1, 5))
@pytest.mark.parametrize(
    "dtype", [np.float64, np.float32, np.int64, np.int32, np.complex64, np.complex128]
)
@pytest.mark.parametrize("shape", [(), (5,), (2, 3)])
def test_keyword_arguments(n_args, dtype, shape):
    @nm.jit
    def fill(**kwargs):
        for i, arg in kwargs.items():
            arg[:] = int(i[3])

    args = {f"in_{i}": np.empty(shape, dtype=dtype) for i in range(n_args)}
    fill(**args)

    for i, arg in args.items():
        expected = np.full(shape, float(i[3]), dtype=dtype)
        np.testing.assert_allclose(arg, expected)


@pytest.mark.parametrize("n_args", range(1, 4))
@pytest.mark.parametrize("n_kwargs", range(1, 4))
def test_variable_positional_and_keyword_arguments(n_args, n_kwargs):
    @nm.jit
    def fill(*args, **kwargs):
        for i, arg in enumerate(args):
            arg[:] = int(i)
        for i, arg in kwargs.items():
            arg[:] = int(i[3])

    args = [np.empty(()) for _ in range(n_args)]
    kwargs = {f"in_{i}": np.empty(()) for i in range(n_kwargs)}
    fill(*args, **kwargs)

    for i, arg in enumerate(args):
        expected = np.full((), float(i))
        np.testing.assert_allclose(arg, expected)
    for i, arg in kwargs.items():
        expected = np.full((), float(i[3]))
        np.testing.assert_allclose(arg, expected)


@pytest.mark.parametrize("n_args", range(1, 4))
@pytest.mark.parametrize("n_kwargs", range(1, 4))
def test_positional_and_variable_positional_and_keyword_arguments(n_args, n_kwargs):
    @nm.jit
    def fill(a, *args, **kwargs):
        for i, arg in enumerate(args):
            arg[:] = int(i)
            a[:] += int(i)
        for i, arg in kwargs.items():
            arg[:] = int(i[3])
            a += int(i[3])

    args = [np.empty(()) for _ in range(n_args)]
    kwargs = {f"in_{i}": np.empty(()) for i in range(n_kwargs)}
    a = np.zeros(())
    fill(a, *args, **kwargs)

    expected_a = np.zeros(())
    for i, arg in enumerate(args):
        expected = np.full((), float(i))
        expected_a += float(i)
        np.testing.assert_allclose(arg, expected)
    for i, arg in kwargs.items():
        expected = np.full((), float(i[3]))
        expected_a += float(i[3])
        np.testing.assert_allclose(arg, expected)
    np.testing.assert_allclose(a, expected_a)


@pytest.mark.parametrize("n_args", range(1, 4))
@pytest.mark.parametrize("n_kwargs", range(1, 4))
def test_positional_and_variable_positional_and_optional_and_keyword_arguments(n_args, n_kwargs):
    @nm.jit
    def fill(a, *args, b=2.0, **kwargs):
        for i, arg in enumerate(args):
            arg[:] = int(i) + b
            a[:] += int(i)
        for i, arg in kwargs.items():
            arg[:] = int(i[3]) + b
            a += int(i[3])
        a[:] -= b

    args = [np.empty(()) for _ in range(n_args)]
    kwargs = {f"in_{i}": np.empty(()) for i in range(n_kwargs)}
    a = np.zeros(())
    fill(a, *args, **kwargs)

    expected_a = np.zeros(())
    for i, arg in enumerate(args):
        expected = np.full((), float(i) + 2.0)
        expected_a += float(i)
        np.testing.assert_allclose(arg, expected)
    for i, arg in kwargs.items():
        expected = np.full((), float(i[3]) + 2.0)
        expected_a += float(i[3])
        np.testing.assert_allclose(arg, expected)
    expected_a -= 2.0
    np.testing.assert_allclose(a, expected_a)


def test_optional_array_argument():
    @nm.jit
    def fill(a, b=np.zeros((5,), dtype=np.float64)):
        for i in nm.range(a.shape[0]):
            a[i] = b[i] + 1.0

    arr = np.empty((5,), dtype=np.float64)
    fill(arr)
    np.testing.assert_allclose(arr, np.ones((5,), dtype=np.float64))

    other = np.full((5,), 2.0, dtype=np.float64)
    fill(arr, other)
    np.testing.assert_allclose(arr, np.full((5,), 3.0, dtype=np.float64))


@pytest.mark.parametrize(
    "order",
    [
        ("a", "b", "c"),
        ("b", "c", "a"),
        ("c", "a", "b"),
    ],
)
def test_all_keyword_permutations(order):
    @nm.jit
    def fill(a, b, c=1.0):
        a[:] = b + c

    a = np.zeros(())
    params = {"a": a, "b": 2.0, "c": 3.0}
    call_kwargs = {k: params[k] for k in order}
    fill(**call_kwargs)

    np.testing.assert_allclose(a, np.full((), 5.0))


def test_optional_keyword_only_comptime_argument():

    @nm.jit
    def fill(a, *, length: nm.comptime = 4):
        for i in nm.range(length):
            a[i] = -i

    a = np.zeros(6, dtype=np.float64)
    fill(a)
    expected = np.zeros_like(a)
    expected[:4] = -np.arange(4, dtype=a.dtype)
    np.testing.assert_allclose(a, expected)

    b = np.zeros(6, dtype=np.float64)
    fill(b, length=5)
    expected = np.zeros_like(b)
    expected[:5] = -np.arange(5, dtype=b.dtype)
    np.testing.assert_allclose(b, expected)


def test_keyword_arguments_have_unique_signature():
    @nm.jit
    def fill(**kwargs):
        for i, arg in kwargs.items():
            arg[:] = float(i[3])

    args = {f"in_{i}": np.zeros(()) for i in range(3)}
    fill(**args)

    for i, arg in args.items():
        np.testing.assert_allclose(arg, float(i[3]))

    args = {f"in_{i}": np.zeros(()) for i in range(2, -1, -1)}
    fill(**args)

    for i, arg in args.items():
        np.testing.assert_allclose(arg, float(i[3]))

    assert len(fill.get_symbolic_functions()) == 1


def test_keyword_arguments_have_unique_signature_unordered():

    from numeta.settings import settings

    settings.unset_reorder_kwargs()

    @nm.jit
    def fill(**kwargs):
        for i, arg in kwargs.items():
            arg[:] = float(i[3])

    args = {f"in_{i}": np.zeros(()) for i in range(3)}
    fill(**args)

    for i, arg in args.items():
        np.testing.assert_allclose(arg, float(i[3]))

    args = {f"in_{i}": np.zeros(()) for i in range(2, -1, -1)}
    fill(**args)

    for i, arg in args.items():
        np.testing.assert_allclose(arg, float(i[3]))

    assert len(fill.get_symbolic_functions()) == 2

    settings.set_reorder_kwargs()
