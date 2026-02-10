from numeta.builder_helper import BuilderHelper
from numeta.settings import settings
from numeta.fortran.external_modules.omp import omp


def prange(*args, **kwargs):
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = None
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = None
    elif len(args) == 3:
        start = args[0]
        stop = args[1]
        step = args[2]
    else:
        raise ValueError("Invalid number of arguments")

    builder = BuilderHelper.get_current_builder()
    I = builder.generate_local_variables("fc_i", dtype=settings.syntax.DEFAULT_INT)

    with omp.parallel_for(I, start, stop - 1, step=step, **kwargs):
        yield I
