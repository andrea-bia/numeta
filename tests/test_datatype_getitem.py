import numeta as nm


def test_datatype_getitem():
    arr = nm.int32[2, 3]
    assert isinstance(arr, nm.datatype.ArrayType)
    assert arr.dtype is nm.int32
    assert arr.shape.dims == (2, 3)

    colon_arr = nm.float64[:]
    assert colon_arr.shape.dims == (None,)
