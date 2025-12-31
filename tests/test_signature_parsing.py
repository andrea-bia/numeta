import numpy as np

from numeta.signature import (
    convert_signature_to_argument_specs,
    get_signature_and_runtime_args,
    parse_function_parameters,
)
from numeta.types_hint import comptime


def test_parse_parameters_with_varargs_and_kwargs():
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


def test_signature_parsing_with_comptime_and_keyword_only():
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


def test_signature_parsing_with_varargs():
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
