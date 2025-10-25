class ExternalLibrary:
    """
    A class to represent an external library.
    It is used to link external libraries to the fortran code.
    Is is child of Module class, where the module is hidden.
    Can contain ExternalModule objects.
    """

    def __init__(
        self,
        name,
        directory=None,
        include=None,
        obj_files=None,
        additional_flags=None,
        to_link=True,
    ):
        """
        Directory is the path to the directory where the external library to link is located.
        Include is the path of the header file to include.
        """
        self.name = name
        self.hidden = True
        self.external = True
        self.directory = directory
        self._include = include
        self._obj_files = obj_files
        self.additional_flags = additional_flags
        self.to_link = to_link

        self.modules = {}
        self.subroutines = {}
        self.variables = {}

    @property
    def obj_files(self):
        return self._obj_files

    @property
    def include(self):
        return self._include
