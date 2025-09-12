from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ArrayShape:
    # None      ⇒ unknown shape
    # ()        ⇒ scalar
    # (3,4)     ⇒ fixed 2-D So it is know at compile time
    # (None, 4) ⇒ 2-D with an undefined dimension at compile time
    _dims: Optional[Tuple]
    fortran_order: bool = False

    def __repr__(self) -> str:
        if self._dims is None:
            return "ArrayShape<unknown>"
        elif len(self._dims) == 0:
            return "ArrayShape<scalar>"
        else:
            inner = ", ".join(map(str, self.dims))
            return f"ArrayShape[{inner}]"

    @property
    def dims(self) -> Tuple:
        if self._dims is None:
            raise ValueError("Cannot get comptime undefined dimensions for unknown shape.")
        return self._dims

    @property
    def rank(self) -> int:
        """
        Returns the rank of the array shape.
        """
        return len(self.dims)

    @property
    def comptime_undefined_dims(self):
        """
        Returns the indices of the dimensions that are undefined at compile time.
        """
        return [i for i, dim in enumerate(self.dims) if isinstance(dim, int)]

    def has_comptime_undefined_dims(self):
        """
        Checks if the argument has undefined dimensions at compile time.
        """
        if self._dims is None:
            # The dimensions are unknown
            return False
        for dim in self.dims:
            if not isinstance(dim, int):
                return True
        return False


# sentinels
UNKNOWN = ArrayShape(None)
SCALAR = ArrayShape(())
