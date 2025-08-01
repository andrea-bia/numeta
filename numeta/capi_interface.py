from string import Template
import subprocess as sp
import sysconfig
import numpy as np


class CAPIInterface:
    def __init__(
        self,
        name,
        module_name,
        args_details,
        directory,
        compile_flags,
        do_checks=True,
        compiler="gcc",
    ):
        self.name = name
        self.module_name = module_name
        self.args_details = args_details
        self.directory = directory
        self.compile_flags = compile_flags
        self.do_checks = do_checks
        self.compiler = compiler

    def generate(self):
        capi_interface = self.construct_module()
        capi_interface_src = self.directory / f"{self.module_name}.c"
        capi_interface_src.write_text(capi_interface)
        capi_obj_path = self.compile(capi_interface_src)
        return capi_obj_path

    def construct_module(self):
        template = """
#include <Python.h>
#include <numpy/arrayobject.h>

void numpy_allocate(void **ptr, size_t *nbytes) {
    *ptr = PyDataMem_NEW(*nbytes);
}

void numpy_deallocate(void **ptr) {
    PyDataMem_FREE(*ptr);
}

${struct_definitions}

${procedure_definitions}

// Method definition table
static PyMethodDef Methods[] = {
    ${module_methods}
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
static struct PyModuleDef ${module_name} = {
    PyModuleDef_HEAD_INIT,
    "${module_name}",  // Module name
    NULL,  // Optional module documentation
    -1,
    Methods
};

// Module initialization function
PyMODINIT_FUNC PyInit_${module_name}(void) {
    import_array();  // Initialize NumPy API
    return PyModule_Create(&${module_name});
}
"""

        substitutions = {}
        substitutions["module_name"] = f"{self.module_name}"

        struct_definitions = {}
        for arg in self.args_details:
            if not arg.is_comptime and arg.datatype.is_struct():
                # bfs to declare all nested structs
                def bfs(struct):
                    for _, nested_dtype, _ in struct.members:
                        if nested_dtype.is_struct() and nested_dtype.name not in struct_definitions:
                            bfs(nested_dtype)
                    struct_definitions[struct.name] = struct.c_declaration()

                bfs(arg.datatype)
        substitutions["struct_definitions"] = "".join(struct_definitions.values())

        module_methods = [
            f'{{"{self.name}", (PyCFunction)(void(*)(void))fc_{self.name}, METH_FASTCALL, "Wrapper generated by fortranic"}},'
        ]
        substitutions["module_methods"] = "".join(module_methods)

        procedure_definitions = [self.construct_procedure()]
        substitutions["procedure_definitions"] = "\n".join(procedure_definitions)

        module_template = Template(template)
        module_template = module_template.substitute(substitutions)

        return module_template

    def construct_procedure(self):
        template = """
void ${fortran_name}(${fortran_args});

static PyObject* ${procedure_name}(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {

    if (nargs != ${n_args}) {
        PyErr_SetString(PyExc_TypeError, "${fortran_name} takes exactly ${n_args} arguments");
        return NULL;
    }

    ${initializations}

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_TypeError, "Error occurred while parsing arguments.");
        return NULL;
    }

    ${checks}

    ${fortran_name}(${call_args});

    Py_RETURN_NONE;
}
"""
        args = [a for a in self.args_details if not a.is_comptime]

        substitutions = {}
        substitutions["fortran_name"] = self.name
        substitutions["procedure_name"] = f"fc_{self.name}"
        substitutions["n_args"] = len(args)

        substitutions["fortran_args"] = ", ".join([self.get_fortran_args(var) for var in args])
        substitutions["initializations"] = "\n    ".join(
            [self.get_initialization(i, var) for i, var in enumerate(args)]
        )

        substitutions["checks"] = ""
        if self.do_checks:
            substitutions["checks"] = "\n    ".join([self.get_check(var) for var in args])

        substitutions["call_args"] = ", ".join([self.get_call_args(var) for var in args])

        return Template(template).substitute(substitutions)

    def get_fortran_args(self, variable):
        """
        Returns the Fortran argument declaration for a given variable.
        Example for this function:

            void matmul(int32_t* a, int32_t* b, int32_t* c);

        this function returns the string 'int32_t* a' for the first argument.
        """

        if variable.to_pass_by_value:
            return f"{variable.datatype.get_cnumpy()} {variable.name}"
        elif variable.shape.has_comptime_undefined_dims():
            return (
                f"npy_intp* {variable.name}_dims, {variable.datatype.get_cnumpy()}* {variable.name}"
            )
        else:
            return f"{variable.datatype.get_cnumpy()}* {variable.name}"

    def get_call_args(self, variable):
        """
        Returns the call argument declaration for a given variable.
        Example for this function:

            ciao((int32_t)a);

        this function returns the string '(int32_t)a' for the first argument.
        """
        if variable.to_pass_by_value:
            return f"({variable.datatype.get_cnumpy()}){variable.name}"
        elif variable.rank == 0 and variable.datatype.is_struct():
            return f"({variable.datatype.get_cnumpy()}*){variable.name}"
        elif variable.shape.has_comptime_undefined_dims():
            return f"PyArray_DIMS({variable.name}), ({variable.datatype.get_cnumpy()}*)PyArray_DATA({variable.name})"
        else:
            return f"({variable.datatype.get_cnumpy()}*)PyArray_DATA({variable.name})"

    def get_initialization(self, i, variable):
        """
        Returns the declaration and initialization of the variable in the C API interface.
        For instance:

            int32_t a = (int32_t)PyLong_AsLongLong(args[0]);

        Where args[idx] is the corresponding argument in the Python function.
        """

        if variable.to_pass_by_value:
            if variable.datatype.get_numpy() in (np.complex64, np.complex128):
                result = "\n"
                result += "#if NPY_ABI_VERSION < 0x02000000\n"
                result += f"    {variable.datatype.get_cnumpy()} {variable.name};\n"
                result += f"    {variable.name}.real = PyComplex_RealAsDouble(args[{i}]);\n"
                result += f"    {variable.name}.imag = PyComplex_RealAsDouble(args[{i}]);\n"
                result += "#else\n"
                cast = variable.datatype.get_capi_cast(f"args[{i}]")
                result += f"    {variable.datatype.get_cnumpy()} {variable.name} = ({variable.datatype.get_cnumpy()}){cast};\n"
                result += "#endif"
                return result
            cast = variable.datatype.get_capi_cast(f"args[{i}]")
            return f"{variable.datatype.get_cnumpy()} {variable.name} = ({variable.datatype.get_cnumpy()}){cast};"
        elif variable.rank == 0 and variable.datatype.is_struct():
            result = f"{variable.datatype.get_cnumpy()}* {variable.name} = NULL;\n"
            result += f"     PyArray_ScalarAsCtype(args[{i}], &{variable.name});"
            return result
        return f"PyArrayObject *{variable.name} = (PyArrayObject*)args[{i}];"

    def get_check(self, variable):
        if variable.rank != 0:
            check_array = f"""
    if (!PyArray_Check({variable.name})) {{
        PyErr_SetString(PyExc_TypeError, "Argument '{variable.name}' must be a NumPy array");
        return NULL;
    }}"""
            if variable.shape.fortran_order:
                check_farray = f"""
    if (!PyArray_ISFARRAY({variable.name})) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' is not Fortran contiguous or aligned.");
        return NULL; 
    }}"""
            else:
                check_farray = f"""
    if (!PyArray_ISCARRAY({variable.name})) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' is not C contiguous or aligned.");
        return NULL; 
    }}"""
            if variable.datatype.is_struct():
                check_align = f"""
    if (!PyDataType_FLAGCHK((PyArray_Descr*)PyArray_DESCR({variable.name}), NPY_ALIGNED_STRUCT)) {{
        PyErr_SetString(PyExc_ValueError, "Input struct {variable.name} dtype is not aligned, use align=True ");
        return NULL;
    }}"""
                return check_array + check_farray + check_align
            else:
                check_type = f"""
    if (PyArray_TYPE({variable.name}) != {variable.datatype.get_cnumpy().upper()}) {{
        PyErr_SetString(PyExc_ValueError, "Input array '{variable.name}' does not have the required type ({variable.datatype.get_numpy()}) "); 
        return NULL;
    }}"""
                return check_array + check_farray + check_type
        else:
            return ""

    def compile(
        self,
        capi_interface_src,
    ):
        """
        Compiles the C API interface.

        Parameters:
            capi_interface_src (str): Source code of the C API interface.

        Returns:
            Path: Path to the compiled C API interface object file.
        """

        include_dirs = [sysconfig.get_paths()["include"], np.get_include()]

        capi_obj_path = self.directory / f"{self.name}_capi.o"

        command = [self.compiler]
        command.extend(self.compile_flags)
        command.extend(["-fPIC", "-c", "-o", str(capi_obj_path), str(capi_interface_src)])
        command.extend([f"-I{inc_dir}" for inc_dir in include_dirs])
        command.extend(["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"])

        try:
            sp.run(command, check=True, cwd=self.directory)
        except sp.CalledProcessError as e:
            print(f"Compilation failed with error: {e}")
            raise

        return capi_obj_path
