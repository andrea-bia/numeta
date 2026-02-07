import numpy as np
import pytest

from numeta.signature import (
    convert_signature_to_argument_specs,
    fast_dispatch,
    get_signature_and_runtime_args,
    get_signature_and_runtime_args_py,
    parse_function_parameters,
    _signature_c_available,
)
from numeta.types_hint import comptime


def test_parse_parameters_with_varargs_and_kwargs(backend):
    def sample(a, *values, b=1, **options):
        return a, values, b, options

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    assert catch_var_positional_name == "values"
    assert n_positional_or_default_args == 2
    assert [params[idx].name for idx in fixed_param_indices] == ["a", "b"]


def test_signature_parsing_with_comptime_and_keyword_only(backend):
    def sample(a: comptime, b, *, c=2):
        return a, b, c

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    array = np.arange(4, dtype=np.float32)
    to_execute, signature, runtime_args = get_signature_and_runtime_args(
        (3, array),
        {"c": 5},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert to_execute is True
    assert signature[0] == 3
    assert signature[1][0] == "b"
    assert signature[1][1] == array.dtype
    assert signature[2][0] == "c"
    assert runtime_args == [array, 5]

    argument_specs = convert_signature_to_argument_specs(
        signature,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
    )
    assert argument_specs[0].is_comptime is True
    assert argument_specs[1].name == "b"
    assert argument_specs[2].name == "c"
    assert argument_specs[2].is_keyword is True


def test_signature_parsing_with_varargs(backend):
    def sample(a, *vals):
        return a, vals

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    vector = np.arange(3, dtype=np.int32)
    to_execute, signature, runtime_args = get_signature_and_runtime_args(
        (1, vector),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert to_execute is True
    assert signature[0] == ("a", int)
    assert signature[1][0] == ("vals", 0)
    assert runtime_args == [1, vector]

    argument_specs = convert_signature_to_argument_specs(
        signature,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
    )
    assert argument_specs[0].name == "a"
    assert argument_specs[1].name == "vals_0"
    assert argument_specs[1].is_keyword is False


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_basic_types(backend):
    """Test that C and Python implementations return identical results for basic types."""

    def sample(a, b, c, d):
        return a, b, c, d

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    # Test with various basic types
    int_arg = 42
    float_arg = 3.14
    complex_arg = 1 + 2j
    array_arg = np.array([1, 2, 3], dtype=np.float64)

    c_result = get_signature_and_runtime_args(
        (int_arg, float_arg, complex_arg, array_arg),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        (int_arg, float_arg, complex_arg, array_arg),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    # to_execute should match
    assert c_result[0] == py_result[0]

    # signatures should match (compare element by element)
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig, f"Signature mismatch: C={c_sig}, Python={py_sig}"

    # runtime_args should match
    assert c_result[2] == py_result[2]


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_numpy_types(backend):
    """Test parity for numpy scalar types."""

    def sample(a, b, c, d, e, f):
        return a, b, c, d, e, f

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    # Test various numpy scalar types
    args = (
        np.int32(1),
        np.int64(2),
        np.float32(3.0),
        np.float64(4.0),
        np.complex64(1 + 2j),
        np.complex128(3 + 4j),
    )

    c_result = get_signature_and_runtime_args(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert c_result[0] == py_result[0]
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_ndarrays(backend):
    """Test parity for numpy arrays with different shapes and dtypes."""

    def sample(a, b, c, d):
        return a, b, c, d

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    # Test arrays with different properties
    args = (
        np.array([1, 2, 3], dtype=np.int32),  # 1D int32
        np.array([[1, 2], [3, 4]], dtype=np.float64),  # 2D float64
        np.array([[[1, 2]]], dtype=np.complex128),  # 3D complex128
        np.asfortranarray(np.array([[1, 2], [3, 4]], dtype=np.float32)),  # Fortran order
    )

    c_result = get_signature_and_runtime_args(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert c_result[0] == py_result[0]
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_kwargs(backend):
    """Test parity for keyword arguments."""

    def sample(a, b=10, c=20):
        return a, b, c

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    args = (np.array([1, 2]),)
    kwargs = {"b": 5, "c": np.float32(3.0)}

    c_result = get_signature_and_runtime_args(
        args,
        kwargs,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        args,
        kwargs,
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert c_result[0] == py_result[0]
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_varargs(backend):
    """Test parity for *args."""

    def sample(a, *rest):
        return a, rest

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    args = (
        np.int64(1),
        np.array([2, 3], dtype=np.float32),
        np.float64(4.0),
    )

    c_result = get_signature_and_runtime_args(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert c_result[0] == py_result[0]
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_c_python_parity_comptime(backend):
    """Test parity with comptime arguments."""

    def sample(n: comptime, arr):
        return n, arr

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    args = (5, np.array([1, 2, 3, 4, 5], dtype=np.float32))

    c_result = get_signature_and_runtime_args(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    py_result = get_signature_and_runtime_args_py(
        args,
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    assert c_result[0] == py_result[0]
    assert len(c_result[1]) == len(py_result[1])
    for c_sig, py_sig in zip(c_result[1], py_result[1]):
        assert c_sig == py_sig
    assert c_result[2] == py_result[2]


# ============================================================================
# fast_dispatch tests
# ============================================================================


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_fast_dispatch_cache_hit(backend):
    """Test fast_dispatch returns the function result on cache hit."""

    def sample(a, b):
        return a, b

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    # First, get the signature so we know the cache key
    _, signature, _ = get_signature_and_runtime_args(
        (a, b),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    # Create a mock fast_call_dict with the expected signature
    def mock_func(*args):
        return sum(x.sum() for x in args)

    fast_call_dict = {signature: mock_func}

    hit, to_execute, payload, runtime_args = fast_dispatch(
        (a, b),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
        fast_call_dict=fast_call_dict,
    )

    assert hit is True
    assert to_execute is True
    assert payload == a.sum() + b.sum()  # 1+2+3+4+5+6 = 21
    assert runtime_args is None


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_fast_dispatch_cache_miss(backend):
    """Test fast_dispatch returns signature tuple on cache miss."""

    def sample(a, b):
        return a, b

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    # Empty dict — always a cache miss
    fast_call_dict = {}

    hit, to_execute, payload, runtime_args = fast_dispatch(
        (a, b),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
        fast_call_dict=fast_call_dict,
    )

    assert hit is False
    assert to_execute is True
    # payload should be the signature tuple
    assert isinstance(payload, tuple)
    assert len(payload) == 2  # Two params
    # runtime_args should be a list with the actual args
    assert len(runtime_args) == 2
    assert np.array_equal(runtime_args[0], a)
    assert np.array_equal(runtime_args[1], b)


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_fast_dispatch_cache_hit_scalars(backend):
    """Test fast_dispatch with scalar args and cache hit."""

    def sample(a, b, c):
        return a, b, c

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    # Get signature for this arg combination
    _, signature, _ = get_signature_and_runtime_args(
        (1, 2.0, 3 + 4j),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    call_count = 0

    def mock_func(*args):
        nonlocal call_count
        call_count += 1
        return 42

    fast_call_dict = {signature: mock_func}

    hit, to_execute, payload, runtime_args = fast_dispatch(
        (1, 2.0, 3 + 4j),
        {},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
        fast_call_dict=fast_call_dict,
    )

    assert hit is True
    assert payload == 42
    assert call_count == 1


@pytest.mark.skipif(not _signature_c_available, reason="C extension not available")
def test_fast_dispatch_signature_matches_get_signature(backend):
    """Verify fast_dispatch cache miss returns the same signature as get_signature_and_runtime_args."""

    def sample(a, b, c=10):
        return a, b, c

    (
        params,
        fixed_param_indices,
        n_positional_or_default_args,
        catch_var_positional_name,
    ) = parse_function_parameters(sample)

    a = np.array([1, 2], dtype=np.int32)
    b = np.float64(3.14)

    _, expected_sig, expected_args = get_signature_and_runtime_args(
        (a, b),
        {"c": np.array([7, 8], dtype=np.float32)},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
    )

    hit, to_execute, sig, runtime_args = fast_dispatch(
        (a, b),
        {"c": np.array([7, 8], dtype=np.float32)},
        params=params,
        fixed_param_indices=fixed_param_indices,
        n_positional_or_default_args=n_positional_or_default_args,
        catch_var_positional_name=catch_var_positional_name,
        fast_call_dict={},  # empty — cache miss
    )

    assert hit is False
    assert to_execute is True
    assert sig == expected_sig
    assert len(runtime_args) == len(expected_args)
