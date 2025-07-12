from numeta.syntax import PointerAssignment
from numeta.builder_helper import BuilderHelper
from numeta.array_shape import ArrayShape


def reshape(variable, shape, order="C"):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"

    if not isinstance(shape, (tuple, list)):
        shape = (shape,)

    if not fortran_order:
        shape = tuple(reversed(shape))

    builder = BuilderHelper.get_current_builder()
    pointer = builder.generate_local_variables(
        "fc_v",
        ftype=variable._ftype,
        shape=ArrayShape(tuple([None for _ in shape]), fortran_order=fortran_order),
        pointer=True,
    )

    PointerAssignment(pointer, shape, variable)

    return pointer
