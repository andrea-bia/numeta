from .syntax import Variable, Scope, Subroutine
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

    @classmethod
    def generate_local_variables(cls, prefix, allocate=False, name=None, **kwargs):
        """
        TODO:
        TO DEPRECATE in some way, use:
        builder = BuilderHelper.get_current_builder()
        bilder.generate_local_variable(...)
        problem: sometimes is not required to have a builder (i.e. fixed name) so maybe set a default one (?).
        """
        if name is None:
            builder = cls.get_current_builder()
            if prefix not in builder.prefix_counter:
                builder.prefix_counter[prefix] = 0
            builder.prefix_counter[prefix] += 1
            name = f"{prefix}{builder.prefix_counter[prefix]}"
        if allocate:
            builder = cls.get_current_builder()
            return builder.allocate_array(name, **kwargs)
        return Variable(name, **kwargs)

    def generate_local_variable(self, prefix, allocate=False, name=None, **kwargs):
        return BuilderHelper.generate_local_variables(
            prefix, allocate=allocate, name=name, **kwargs
        )

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
        variable_ptr = Variable(f"{name}_c_ptr", FPointer_c)
        self.allocated_arrays[name] = variable_ptr

        dtype = DataType.from_ftype(kwargs["ftype"])

        size = dtype.get_nbytes()
        for dim in shape.dims:
            size *= dim

        # allocate memory with the numpy allocator
        numpy_mem.numpy_allocate(variable_ptr, size)

        # Fortran is so versone
        # create fortran pointer (with lower bound 1)
        variable_lb1 = Variable(
            f"{name}_f_ptr_lb1", ftype=kwargs["ftype"], shape=ArrayShape((None,)), pointer=True
        )
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

    def build(self, *args, **kwargs):
        old_builder = self.current_builder
        self.set_current_builder(self)

        old_scope = Scope.current_scope
        self.symbolic_function.scope.enter()

        return_variables = self.numeta_function.run_symbolic(*args, **kwargs)

        if return_variables is None:
            return_variables = []
        elif not isinstance(return_variables, (list, tuple)):
            return_variables = [return_variables]

        from .array_shape import ArrayShape, SCALAR, UNKNOWN
        from .syntax import Shape
        from .datatype import DataType, size_t

        ret = []
        for i, var in enumerate(return_variables):
            expr = None
            if isinstance(var, Variable):
                if var.name in self.allocated_arrays:

                    rank = var._shape.rank
                    shape = Variable(
                        f"fc_out_shape_{i}",
                        ftype=size_t.get_fortran(bind_c=True),
                        shape=ArrayShape((rank,)),
                        intent="out",
                    )
                    self.symbolic_function.add_variable(shape)
                    # add to the symbolic function
                    shape[:] = Shape(var)
                    shape[:] = shape[rank - 1 : 1 : -1]  # reverse the shape for Fortran order

                    ptr = self.allocated_arrays.pop(var.name)
                    ptr.intent = "out"
                    self.symbolic_function.add_variable(ptr)

                    ret.append((DataType.from_ftype(var._ftype), rank))
                elif var._shape is SCALAR and var.name not in self.symbolic_function.arguments:

                    var.intent = "out"
                    self.symbolic_function.add_variable(var)

                    ret.append((DataType.from_ftype(var._ftype), 0))
                else:
                    if var._shape is SCALAR:
                        tmp = BuilderHelper.generate_local_variables("fc_s", ftype=var._ftype)
                        tmp[:] = var
                        tmp.intent = "out"
                        self.symbolic_function.add_variable(tmp)
                        ret.append((DataType.from_ftype(var._ftype), 0))
                        continue
                    expr = var
            else:
                # it is an expression
                expr = var

            if expr is not None:
                # We have to copy the expression in a new array
                expr_shape = expr._shape
                if expr_shape is SCALAR:
                    tmp = BuilderHelper.generate_local_variables("fc_s", ftype=expr._ftype)
                    tmp[:] = expr
                    tmp.intent = "out"
                    self.symbolic_function.add_variable(tmp)
                    ret.append((DataType.from_ftype(expr._ftype), 0))
                    continue

                if expr_shape is UNKNOWN:
                    raise NotImplementedError(
                        "Returning arrays with unknown shape is not supported yet."
                    )

                rank = expr_shape.rank
                shape = Variable(
                    f"fc_out_shape_{i}",
                    ftype=size_t.get_fortran(bind_c=True),
                    shape=ArrayShape((rank,)),
                    intent="out",
                )
                self.symbolic_function.add_variable(shape)

                # Cache the expression shape in a local array so we can reuse it
                # both for allocating the temporary buffer and to report the
                # extents back through the C API.
                tmp_shape = Variable(
                    f"fc_out_shape_{i}_tmp",
                    ftype=size_t.get_fortran(bind_c=True),
                    shape=ArrayShape((rank,)),
                )
                tmp_shape[:] = Shape(expr)

                alloc_dims = [tmp_shape[i] for i in range(rank)]
                if not expr_shape.fortran_order:
                    alloc_dims = alloc_dims[::-1]

                from .wrappers import empty

                tmp = empty(
                    alloc_dims,
                    dtype=expr._ftype,
                    order="F" if expr_shape.fortran_order else "C",
                )

                shape[:] = tmp_shape
                # Reverse the shape for the C interface after materializing the
                # temporary buffer so the allocation happens with the original
                # extents.
                shape[:] = shape[rank - 1 : 1 : -1]
                tmp[:] = expr

                ptr = self.allocated_arrays.pop(tmp.name)
                ptr.intent = "out"
                self.symbolic_function.add_variable(ptr)

                ret.append((DataType.from_ftype(expr._ftype), rank))

        for array in self.allocated_arrays.values():
            self.deallocate_array(array)

        self.symbolic_function.scope.exit()
        Scope.current_scope = old_scope

        self.set_current_builder(old_builder)

        return ret

    def inline(self, function, *arguments):
        """Inline ``function`` with the given ``arguments`` into the current scope."""
        # Avoid heavy imports at module load time
        if not isinstance(function, Subroutine):
            raise TypeError("Unsupported function type for inline call")

        from .syntax.tools import check_node

        args = [check_node(arg) for arg in arguments]
        if len(args) != len(function.arguments):
            raise ValueError("Incorrect number of arguments for inlined subroutine")
        variables_couples = list(zip(function.arguments.values(), args))
        for local_variable in function.get_local_variables().values():
            new_local_variable = self.generate_local_variables(
                "nm_inline_",
                ftype=local_variable._ftype,
                shape=local_variable._shape,
                intent=local_variable.intent,
                pointer=local_variable.pointer,
                target=local_variable.target,
                allocatable=local_variable.allocatable,
                parameter=local_variable.parameter,
                assign=local_variable.assign,
                bind_c=local_variable.bind_c,
            )
            variables_couples.append((local_variable, new_local_variable))

        for stmt in function.scope.get_statements():
            Scope.add_to_current_scope(stmt.get_with_updated_variables(variables_couples))
