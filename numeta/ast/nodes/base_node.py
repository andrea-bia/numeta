from abc import ABC, abstractmethod
import inspect


class Node(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Capture source location from where this node was created
        self._source_location = self._capture_source_location()

    def _capture_source_location(self):
        """Capture the source location (filename, line) where this node was created."""
        import os

        try:
            frame = inspect.currentframe()
            # Go up the stack to find the user's code (skip numeta internal frames)
            while frame:
                filename = frame.f_code.co_filename
                # Skip numeta's internal files - check if it's in the numeta package
                # by looking for numeta/ast/, numeta/wrappers/, etc.
                is_numeta_internal = (
                    "numeta/ast/" in filename
                    or "numeta/wrappers/" in filename
                    or "numeta/builder_helper.py" in filename
                    or "numeta/numeta_function.py" in filename
                    or "numeta/ir/" in filename
                    or "numeta/c/" in filename
                    or "numeta/fortran/" in filename
                )
                if not is_numeta_internal:
                    return {
                        "filename": filename,
                        "lineno": frame.f_lineno,
                        "function": frame.f_code.co_name,
                    }
                frame = frame.f_back
        except Exception:
            pass
        return None

    @property
    def source_location(self):
        """Get the source location where this node was created."""
        return self._source_location

    @abstractmethod
    def extract_entities(self):
        """Extract the nested entities of the node."""
