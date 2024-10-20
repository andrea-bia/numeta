from numeta.syntax import PointerAssignment
from numeta.builder_helper import BuilderHelper


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
        ftype=variable.ftype,
        dimension=tuple([None for _ in shape]),
        pointer=True,
        fortran_order=fortran_order,
    )

    PointerAssignment(pointer, shape, variable)

    return pointer
