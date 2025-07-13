from .syntax import Variable, Scope
from .settings import settings


class BuilderHelper:
    current_builder = None

    @classmethod
    def set_current_builder(cls, builder):
        cls.current_builder = builder

    @classmethod
    def get_current_builder(cls):
        if cls.current_builder is None:
            raise Warning("The current builder is not initialized")
        return cls.current_builder

    def __init__(self, numeta_function, symbolic_function, signature):
        self.numeta_function = numeta_function
        self.symbolic_function = symbolic_function
        self.signature = signature

        self.prefix_counter = {}
        self.allocated_arrays = {}

        if settings.use_numpy_allocator:
            self.allocate_array = self._allocate_array_numpy
            self.deallocate_array = self._deallocate_array_numpy 
        else:
            self.allocate_array = self._allocate_array
            self.deallocate_array = self._deallocate_array

    def generate_local_variables(self, prefix, allocate=False, **kwargs):
        if prefix not in self.prefix_counter:
            self.prefix_counter[prefix] = 0
        self.prefix_counter[prefix] += 1
        if allocate:
            return self.allocate_array(f"{prefix}{self.prefix_counter[prefix]}", **kwargs)
        return Variable(f"{prefix}{self.prefix_counter[prefix]}", **kwargs)

    def _allocate_array(self, name, shape, **kwargs):
        from .syntax import Allocate, If, Allocated, Not
        from .array_shape import ArrayShape
        alloc_shape = ArrayShape(tuple([None] * shape.rank), fortran_order=shape.fortran_order) 
        variable = Variable(name, shape=alloc_shape, allocatable=True, **kwargs)
        with If(Not(Allocated(variable))):
            Allocate(variable, *shape.dims)
        self.allocated_arrays[name] = variable
        return variable

    def _deallocate_array(self, array):
        from numeta.syntax import Deallocate, If, Allocated
        with If(Allocated(array)):
            Deallocate(array)

    def _allocate_array_numpy(self, name, shape, **kwargs):
        from .syntax import PointerAssignment
        from .syntax.expressions import ArrayConstructor
        from .wrappers import numpy_mem
        from .external_modules.iso_c_binding import FPointer_c, iso_c
        from .array_shape import ArrayShape
        from .datatype import DataType 

        # create a c pointer variable that will be also deallocated
        variable_ptr =  Variable(f"{name}_c_ptr", FPointer_c)
        self.allocated_arrays[name] = variable_ptr

        dtype = DataType.from_ftype(kwargs['ftype'])
        
        size = dtype.get_nbytes()
        for dim in shape.dims:
            size *= dim

        # allocate memory with the numpy allocator
        numpy_mem.numpy_allocate(variable_ptr, size)

        # Fortran is so versone
        # create fortran pointer (with lower bound 1)
        variable_lb1 = Variable(f"{name}_f_ptr_lb1", ftype=kwargs['ftype'], shape=ArrayShape((None,)), pointer=True)
        # point the fortran pointer to the allocated memory
        iso_c.c_f_pointer(variable_ptr, variable_lb1, ArrayConstructor(size))

        alloc_shape = ArrayShape(tuple([None] * shape.rank), fortran_order=shape.fortran_order) 
        variable = Variable(name, shape=alloc_shape, pointer=True, **kwargs)

        # assign the fortran pointer with the proper lower bound
        PointerAssignment(variable, shape, variable_lb1)

        return variable

    def _deallocate_array_numpy(self, array):
        from .wrappers import numpy_mem
        numpy_mem.numpy_deallocate(array)

    def build(self, *args):
        old_builder = self.current_builder
        self.set_current_builder(self)

        old_scope = Scope.current_scope
        self.symbolic_function.scope.enter()

        self.numeta_function.run_symbolic(*args)

        for array in self.allocated_arrays.values():
            self.deallocate_array(array)

        self.symbolic_function.scope.exit()
        Scope.current_scope = old_scope

        self.set_current_builder(old_builder)
