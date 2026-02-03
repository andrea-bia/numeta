from numeta.ast import Switch, Case


def cases(select, cases_range: range):
    if not isinstance(cases_range, range):
        raise ValueError("The second argument must be a range object")

    with Switch(select):
        for c in cases_range:
            with Case(c):
                yield c
