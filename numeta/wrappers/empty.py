from numeta.builder_helper import BuilderHelper
from numeta.types_hint import float64
from numeta.syntax import Allocate, If, Allocated, Not


def empty(shape, dtype=float64):
    if isinstance(shape, (tuple, list)):
        dimension = tuple([None for _ in range(len(shape))])
    else:
        dimension = (None,)
        shape = (shape,)

    builder = BuilderHelper.get_current_builder()
    array = builder.generate_local_variables(
        "fc_a",
        ftype=dtype.dtype.get_fortran(),
        dimension=dimension,
        allocatable=True,
        fortran_order=False,
    )
    with If(Not(Allocated(array))):
        Allocate(array, *shape)

    return array
