from numeta.builder_helper import BuilderHelper
from numeta.syntax import Do
from numeta.settings import syntax_settings


def range(*args):
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
    I = builder.generate_local_variables("fc_i", ftype=syntax_settings.DEFAULT_INTEGER)

    with Do(I, start, stop - 1, step=step):
        yield I
