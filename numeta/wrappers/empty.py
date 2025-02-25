from numeta.builder_helper import BuilderHelper
from numeta.types_hint import float64
from numeta.syntax import Allocate, If, Allocated, Not, FortranType


def empty(shape, dtype=float64, order="C"):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    
    if isinstance(dtype, FortranType):
        ftype = dtype
    else:
        ftype = dtype.dtype.get_fortran()

    builder = BuilderHelper.get_current_builder()
    array = builder.generate_local_variables(
        "fc_a",
        ftype=ftype,
        dimension=tuple(None for _ in shape),
        allocatable=True,
        fortran_order=fortran_order,
    )
    with If(Not(Allocated(array))):
        Allocate(array, *shape)

    return array
