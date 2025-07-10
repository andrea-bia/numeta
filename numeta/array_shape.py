from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class ArrayShape:
    # None      ⇒ unknown shape
    # ()        ⇒ scalar
    # (3,4)     ⇒ fixed 2-D So it is know at compile time
    # (None, 4) ⇒ 2-D with an undefined dimension at compile time
    dims: Optional[Tuple]

    def __repr__(self) -> str:
        if self.dims is None:
            return "ArrayShape<unknown>"
        elif len(self.dims) == 0:
            return "ArrayShape<scalar>"
        else:
            inner = ", ".join(map(str, self.dims))
            return f"ArrayShape[{inner}]"

    @property
    def comptime_undefined_dims(self):
        """
        Returns the indices of the dimensions that are undefined at compile time.
        """
        if self.dims is None:
            raise ValueError("Cannot get comptime undefined dimensions for unknown shape.")
        return [i for i, dim in enumerate(self.dims) if isinstance(dim, int)]

    def has_comptime_undefined_dims(self):
        """
        Checks if the argument has undefined dimensions at compile time.
        """
        if self.dims is None:
            raise ValueError("Cannot get comptime undefined dimensions for unknown shape.")
        for dim in self.dims:
            if not isinstance(dim, int):
                return True
        return False


# sentinels
UNKNOWN = ArrayShape(None)
SCALAR = ArrayShape(())
