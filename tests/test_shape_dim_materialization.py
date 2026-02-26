from pathlib import Path
import re

import numpy as np
import numeta as nm


def _read_generated_source(tmp_path: Path, func_name: str, backend: str) -> str:
    suffix = "f90" if backend == "fortran" else "c"
    src_path = tmp_path / f"{func_name}_src.{suffix}"
    return src_path.read_text()


def test_non_trivial_empty_shape_materializes_dim_temps(tmp_path, backend):
    func_name = f"shape_materialized_{backend}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def get_shape(a, b, out_shape):
        c = nm.empty((b.shape[0] + 10, b.shape[1] + a.shape[1]), dtype=nm.f8)
        out_shape[:] = nm.Shape(c)

    a = np.ones((3, 4), dtype=np.float64)
    b = np.ones((5, 6), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)

    get_shape(a, b, out_shape)

    expected = np.array([15, 10], dtype=np.int64)
    np.testing.assert_array_equal(out_shape, expected)

    source = _read_generated_source(tmp_path, func_name, backend)
    assert "nm_shape1" in source
    if backend == "fortran":
        assert re.search(r"(?<![A-Za-z0-9_])shape\(", source.lower()) is None
        assert re.search(r"(?<![A-Za-z0-9_])size\(", source.lower()) is None


def test_trivial_empty_shape_does_not_materialize_dim_temps(tmp_path, backend):
    func_name = f"shape_not_materialized_{backend}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def get_shape(b, out_shape):
        c = nm.empty((b.shape[0], b.shape[1]), dtype=nm.f8)
        out_shape[:] = nm.Shape(c)

    b = np.ones((5, 6), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)

    get_shape(b, out_shape)

    expected = np.array([5, 6], dtype=np.int64)
    np.testing.assert_array_equal(out_shape, expected)

    source = _read_generated_source(tmp_path, func_name, backend)
    assert "fc_dim" not in source


def test_non_trivial_reshape_shape_materializes_dim_temps(tmp_path, backend):
    func_name = f"reshape_shape_materialized_{backend}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def get_shape(a, b, out_shape):
        c = nm.reshape(a, (a.shape[0] + b.shape[0] - b.shape[0], a.shape[1]))
        out_shape[:] = nm.Shape(c)

    a = np.ones((5, 6), dtype=np.float64)
    b = np.ones((5,), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)

    get_shape(a, b, out_shape)

    source = _read_generated_source(tmp_path, func_name, backend)
    assert "nm_shape1" in source


def test_trivial_reshape_shape_does_not_materialize_dim_temps(tmp_path, backend):
    func_name = f"reshape_shape_not_materialized_{backend}"

    @nm.jit(backend=backend, directory=str(tmp_path), namer=lambda *spec: func_name)
    def get_shape(a, out_shape):
        c = nm.reshape(a, (a.shape[0], a.shape[1]))
        out_shape[:] = nm.Shape(c)

    a = np.ones((5, 6), dtype=np.float64)
    out_shape = np.zeros(2, dtype=np.int64)

    get_shape(a, out_shape)

    source = _read_generated_source(tmp_path, func_name, backend)
    assert "nm_shape" not in source
