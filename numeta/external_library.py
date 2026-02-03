class ExternalLibrary:
    """
    A class to represent an external library.
    It is used to link external libraries to the generated code.
    It is a child of the Namespace class, where the namespace is hidden.
    Can contain ExternalNamespace objects.
    """

    def __init__(
        self,
        name,
        path=None,
        include=None,
        obj_files=None,
        rpath=None,
        additional_flags=None,
        to_link=True,
    ):
        """
        path is the path to the directory where the external library to link is located.
        Include is the path of the header file to include.
        """
        self.name = name
        self.hidden = True
        self.external = True
        self._path = path
        self._rpath = rpath
        self._include = include
        self._obj_files = obj_files
        self.additional_flags = additional_flags
        self.to_link = to_link

        self.namespaces = {}
        self.procedures = {}
        self.variables = {}

    @property
    def obj_files(self):
        return self._obj_files

    @property
    def include(self):
        return self._include

    @property
    def path(self):
        return self._path

    @property
    def rpath(self):
        return self._rpath
