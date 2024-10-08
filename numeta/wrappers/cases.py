from numeta.syntax import SelectCase, Case


def cases(select, cases_range: range):
    if not isinstance(cases_range, range):
        raise ValueError("The second argument must be a range object")

    with SelectCase(select):
        for c in cases_range:
            with Case(c):
                yield c
