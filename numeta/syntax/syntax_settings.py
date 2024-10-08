class SyntaxSettings:
    def __init__(self):
        self.DEFAULT_INTEGER_KIND = 8
        self.DEFAULT_REAL_KIND = 8
        self.DEFAULT_COMPLEX_KIND = 8
        self.DEFAULT_LOGICAL_KIND = 1
        self.DEFAULT_CHARACTER_KIND = 1
        self.__subroutine_bind_c = False
        self.__derived_type_bind_c = False
        self.__array_lower_bound = 1
        self.__force_value = False  # force the value attribute when intent(in)

    def set_default_integer_kind(self, kind):
        self.DEFAULT_INTEGER_KIND = kind

    def set_default_real_kind(self, kind):
        self.DEFAULT_REAL_KIND = kind

    def set_default_complex_kind(self, kind):
        self.DEFAULT_COMPLEX_KIND = kind

    def set_default_logical_kind(self, kind):
        self.DEFAULT_LOGICAL_KIND = kind

    def set_default_character_kind(self, kind):
        self.DEFAULT_CHARACTER_KIND = kind

    @property
    def subroutine_bind_c(self):
        return self.__subroutine_bind_c

    def set_subroutine_bind_c(self):
        self.__subroutine_bind_c = True

    def unset_subroutine_bind_c(self):
        self.__subroutine_bind_c = False

    @property
    def derived_type_bind_c(self):
        return self.__derived_type_bind_c

    def set_derived_type_bind_c(self):
        self.__derived_type_bind_c = True

    def unset_derived_type_bind_c(self):
        self.__derived_type_bind_c = False

    @property
    def array_lower_bound(self):
        return self.__array_lower_bound

    def set_array_lower_bound(self, value):
        try:
            self.__array_lower_bound = int(value)
        except ValueError as e:
            raise ValueError(f"Array lower bound must be an integer, got {value}")

    @property
    def force_value(self):
        return self.__force_value

    def set_force_value(self):
        self.__force_value = True

    def unset_force_value(self):
        self.__force_value = False


settings = SyntaxSettings()
