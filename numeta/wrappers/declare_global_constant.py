import numpy as np
from numeta.syntax import Variable, Module
from numeta.datatype import DataType, float64, FortranType
from numeta.array_shape import ArrayShape
from numeta.numeta_function import NumetaCompilationTarget

_n_global_constant = 0


def declare_global_constant(
    shape,
    dtype: DataType | FortranType | np.generic = float64,
    order="C",
    name=None,
    value=None,
    directory=None,
):
    if order not in ["C", "F"]:
        raise ValueError(f"Invalid order: {order}, must be 'C' or 'F'")

    fortran_order = order == "F"
    if not isinstance(shape, ArrayShape):
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = ArrayShape(shape, fortran_order=fortran_order)

    if isinstance(dtype, FortranType):
        ftype = dtype
    elif isinstance(dtype, type):
        if issubclass(dtype, DataType):
            ftype = dtype.get_fortran()
        elif issubclass(dtype, np.generic):
            ftype = DataType.from_np_dtype(dtype).get_fortran()
        else:
            raise TypeError(f"Unsupported dtype class: {dtype}")
    else:
        raise TypeError(f"Expected a numpy or numeta dtype got {type(dtype).__name__}")

    if name is None:
        global _n_global_constant
        name = f"global_constant_{_n_global_constant}"
        _n_global_constant += 1

    # Lets create a module to host the global_constant variable, it will be a module variable
    global_constant_var_module = Module(f"{name}_module")

    var = Variable(
        name=name,
        ftype=ftype,
        shape=shape,
        assign=value,
        # TODO
        # parameter=True, # parameter is not supported yet, so not really constant.
        parent=global_constant_var_module,
    )

    # We have to compile the module when needed
    module_library = NumetaCompilationTarget(
        f"{name}_module",
        global_constant_var_module,
        directory=directory,
    )
    global_constant_var_module.parent = module_library
    return var
