from __future__ import annotations
from typing import Any, Iterator

from numeta.ast import Switch, Case


def cases(select: Any, cases_range: range) -> Iterator[int]:
    if not isinstance(cases_range, range):
        raise ValueError("The second argument must be a range object")

    with Switch(select):
        for c in cases_range:
            with Case(c):
                yield c
