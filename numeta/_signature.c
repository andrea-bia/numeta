#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

// Globals for types
static PyObject *ArrayType = NULL;
static PyObject *DataType = NULL;
static PyObject *ExpressionNode = NULL;
static PyObject *Variable = NULL;
static PyObject *GetAttr = NULL;
static PyObject *GetItem = NULL;
static PyObject *SCALAR = NULL;
static PyObject *UNKNOWN = NULL;
static PyObject *INSPECT_EMPTY = NULL;
static PyObject *NumpyGeneric = NULL;

// Globals for strings - interned for speed
static PyObject *str_kind = NULL;
static PyObject *str_name = NULL;
static PyObject *str_default = NULL;
static PyObject *str_is_comptime = NULL;
static PyObject *str_dtype = NULL;
static PyObject *str_names = NULL;
static PyObject *str_shape = NULL;
static PyObject *str_get_numpy = NULL;
static PyObject *str_has_comptime_undefined_dims = NULL;
static PyObject *str_fortran_order = NULL;
static PyObject *str_rank = NULL;
static PyObject *str_dims = NULL;
static PyObject *str_variable = NULL;
static PyObject *str_intent = NULL;
static PyObject *str_in = NULL;
static PyObject *str_inout = NULL;
static PyObject *str_underscore_shape = NULL;

// Constants
static int KIND_POSITIONAL_ONLY = 0;
static int KIND_POSITIONAL_OR_KEYWORD = 1;
static int KIND_VAR_POSITIONAL = 2;
static int KIND_KEYWORD_ONLY = 3;
static int KIND_VAR_KEYWORD = 4;

typedef struct {
    int add_shape_descriptors;
    int ignore_fixed_shape_in_nested_calls;
    int reorder_kwargs;
} Settings;

// Optimized instance check - inline and use fast path
static inline int is_instance_fast(PyObject *obj, PyObject *cls) {
    if (cls == NULL || obj == NULL) return 0;
    // Fast path: check if cls is a type and use PyObject_TypeCheck
    if (PyType_Check(cls)) {
        return PyObject_TypeCheck(obj, (PyTypeObject*)cls);
    }
    int result = PyObject_IsInstance(obj, cls);
    // Clear any exception that might have been set during the isinstance check
    if (result == -1) {
        PyErr_Clear();
        return 0;
    }
    return result;
}

// Fast path for numpy arrays - aggressively inlined
static inline PyObject* sig_from_ndarray(PyObject *arg, PyObject *name) {
    PyArrayObject *arr = (PyArrayObject*)arg;
    PyObject *dtype = (PyObject*)PyArray_DESCR(arr);
    int ndim = PyArray_NDIM(arr);
    int is_fortran = PyArray_ISFORTRAN(arr);
    
    // Use PyTuple_Pack for small fixed-size tuples - faster and cleaner
    PyObject *sig = PyTuple_Pack(4, name, dtype, PyLong_FromLong(ndim), 
                                  is_fortran ? Py_True : Py_False);
    return sig;
}

// Fast path for scalars - aggressively inlined
static inline PyObject* sig_from_scalar(PyObject *arg, PyObject *name) {
    PyObject *type = (PyObject*)Py_TYPE(arg);
    return PyTuple_Pack(2, name, type);
}

// Batch fetch all shape-related attributes at once for ArrayType
// Returns: 0 on success, -1 on error
static int fetch_shape_attrs(PyObject *shape, 
                              PyObject **rank_out, PyObject **fortran_out, 
                              PyObject **dims_out, int *has_comptime_out,
                              int *shape_is_unknown_out, Settings *settings) {
    *shape_is_unknown_out = (shape == UNKNOWN);
    
    // Check for comptime dims
    PyObject *res = PyObject_CallMethodObjArgs(shape, str_has_comptime_undefined_dims, NULL);
    if (res) {
        *has_comptime_out = PyObject_IsTrue(res);
        Py_DECREF(res);
    } else {
        PyErr_Clear();
        *has_comptime_out = 0;
    }
    
    // Fetch all attributes at once
    *rank_out = PyObject_GetAttr(shape, str_rank);
    *fortran_out = PyObject_GetAttr(shape, str_fortran_order);
    *dims_out = PyObject_GetAttr(shape, str_dims);
    
    return 0;
}

// Optimized ArrayType handler with batched attribute lookups
static PyObject* sig_from_arraytype(PyObject *arg, PyObject *name, 
                                     int *to_execute, Settings *settings) {
    *to_execute = 0;
    
    // Get shape and dtype
    PyObject *shape = PyObject_GetAttr(arg, str_shape);
    if (!shape) return NULL;
    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) { Py_DECREF(shape); return NULL; }
    
    PyObject *numpy_dtype = PyObject_CallMethodObjArgs(dtype, str_get_numpy, NULL);
    if (!numpy_dtype) { Py_DECREF(shape); Py_DECREF(dtype); return NULL; }
    
    // Batch fetch all shape attributes
    PyObject *rank, *fortran_order, *dims;
    int has_comptime, shape_is_unknown;
    fetch_shape_attrs(shape, &rank, &fortran_order, &dims, 
                      &has_comptime, &shape_is_unknown, settings);
    
    PyObject *sig;
    
    // Flattened conditionals for better branch prediction
    if (shape_is_unknown || (!settings->add_shape_descriptors && has_comptime)) {
        // Shape unknown or comptime dims without descriptors
        sig = PyTuple_New(4);
        if (!sig) goto cleanup;
        Py_INCREF(name);
        PyTuple_SET_ITEM(sig, 0, name);
        PyTuple_SET_ITEM(sig, 1, numpy_dtype);
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(sig, 2, Py_None);
        if (fortran_order) PyTuple_SET_ITEM(sig, 3, fortran_order);
        else { Py_INCREF(Py_False); PyTuple_SET_ITEM(sig, 3, Py_False); }
    } 
    else if (has_comptime) {
        // Has comptime dims with descriptors
        sig = PyTuple_New(4);
        if (!sig) goto cleanup;
        Py_INCREF(name);
        PyTuple_SET_ITEM(sig, 0, name);
        PyTuple_SET_ITEM(sig, 1, numpy_dtype);
        if (rank) PyTuple_SET_ITEM(sig, 2, rank);
        else { Py_INCREF(Py_None); PyTuple_SET_ITEM(sig, 2, Py_None); }
        if (fortran_order) PyTuple_SET_ITEM(sig, 3, fortran_order);
        else { Py_INCREF(Py_False); PyTuple_SET_ITEM(sig, 3, Py_False); }
    } 
    else {
        // Full shape info available
        sig = PyTuple_New(6);
        if (!sig) goto cleanup;
        Py_INCREF(name);
        PyTuple_SET_ITEM(sig, 0, name);
        PyTuple_SET_ITEM(sig, 1, numpy_dtype);
        if (rank) PyTuple_SET_ITEM(sig, 2, rank);
        else { Py_INCREF(Py_None); PyTuple_SET_ITEM(sig, 2, Py_None); }
        if (fortran_order) PyTuple_SET_ITEM(sig, 3, fortran_order);
        else { Py_INCREF(Py_False); PyTuple_SET_ITEM(sig, 3, Py_False); }
        Py_INCREF(str_inout);
        PyTuple_SET_ITEM(sig, 4, str_inout);
        if (dims) PyTuple_SET_ITEM(sig, 5, dims);
        else { Py_INCREF(Py_None); PyTuple_SET_ITEM(sig, 5, Py_None); }
    }
    
cleanup:
    Py_DECREF(shape);
    Py_DECREF(dtype);
    Py_XDECREF(rank);
    Py_XDECREF(fortran_order);
    Py_XDECREF(dims);
    return sig;
}

// Optimized numpy scalar handler
static inline PyObject* sig_from_numpy_scalar(PyObject *arg, PyObject *name) {
    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) return NULL;
    
    PyObject *names = PyObject_GetAttr(dtype, str_names);
    if (!names) { Py_DECREF(dtype); return NULL; }
    
    PyObject *sig;
    if (names != Py_None) {
        // Structured dtype
        sig = PyTuple_Pack(3, name, dtype, PyLong_FromLong(0));
    } else {
        // Simple dtype
        sig = PyTuple_Pack(2, name, dtype);
    }
    Py_DECREF(names);
    return sig;
}

// Fast path for intent checking - cache the check
static inline PyObject* get_intent_str(PyObject *arg) {
    if (!is_instance_fast(arg, Variable) && 
        !is_instance_fast(arg, GetAttr) && 
        !is_instance_fast(arg, GetItem)) {
        return str_in;
    }
    
    PyObject *target = arg;
    if (is_instance_fast(arg, GetAttr) || is_instance_fast(arg, GetItem)) {
        target = PyObject_GetAttr(arg, str_variable);
        if (!target) {
            PyErr_Clear();  // Clear exception before returning default
            return str_in;
        }
    } else {
        Py_INCREF(target);
    }
    
    PyObject *intent_obj = PyObject_GetAttr(target, str_intent);
    Py_DECREF(target);
    
    if (!intent_obj) {
        PyErr_Clear();  // Clear exception before returning default
        return str_in;
    }
    
    PyObject *result = str_in;
    int cmp_result = PyUnicode_Compare(intent_obj, str_in);
    if (cmp_result == -1 && PyErr_Occurred()) {
        PyErr_Clear();
    } else if (cmp_result != 0) {
        result = str_inout;
    }
    Py_DECREF(intent_obj);
    return result;
}

// Optimized ExpressionNode handler
static PyObject* sig_from_expression_node(PyObject *arg, PyObject *name,
                                          int *to_execute, Settings *settings) {
    *to_execute = 0;
    
    PyObject *dtype = PyObject_GetAttr(arg, str_dtype);
    if (!dtype) {
        return NULL;
    }
    
    PyObject *numpy_dtype = PyObject_CallMethodObjArgs(dtype, str_get_numpy, NULL);
    if (!numpy_dtype) { 
        Py_DECREF(dtype); 
        return NULL; 
    }
    
    PyObject *shape = PyObject_GetAttr(arg, str_underscore_shape);
    if (!shape) { 
        Py_DECREF(dtype); 
        Py_DECREF(numpy_dtype); 
        return NULL; 
    }
    
    PyObject *intent_str = get_intent_str(arg);
    
    int shape_is_scalar = (shape == SCALAR);
    int shape_is_unknown = (shape == UNKNOWN);
    
    int has_comptime_dims = 0;
    // Only check for comptime dims if not scalar or unknown
    if (!shape_is_scalar && !shape_is_unknown) {
        PyObject *res = PyObject_CallMethodObjArgs(shape, str_has_comptime_undefined_dims, NULL);
        if (res) { 
            has_comptime_dims = PyObject_IsTrue(res); 
            Py_DECREF(res); 
        } else {
            PyErr_Clear();
        }
    }
    
    PyObject *sig = NULL;
    
    if (shape_is_scalar) {
        if (intent_str == str_inout) {
            sig = PyTuple_Pack(5, name, numpy_dtype, PyLong_FromLong(0), 
                              Py_False, intent_str);
        } else {
            sig = PyTuple_Pack(2, name, numpy_dtype);
        }
    }
    else if (shape_is_unknown || (!settings->add_shape_descriptors && has_comptime_dims)) {
        sig = PyTuple_Pack(5, name, numpy_dtype, Py_None, Py_False, intent_str);
    }
    else if (has_comptime_dims) {
        PyObject *rank = PyObject_GetAttr(shape, str_rank);
        PyObject *fortran_order = PyObject_GetAttr(shape, str_fortran_order);
        sig = PyTuple_New(5);
        if (sig) {
            Py_INCREF(name);
            PyTuple_SET_ITEM(sig, 0, name);
            PyTuple_SET_ITEM(sig, 1, numpy_dtype);
            PyTuple_SET_ITEM(sig, 2, rank ? rank : Py_None);
            PyTuple_SET_ITEM(sig, 3, fortran_order ? fortran_order : Py_False);
            Py_INCREF(intent_str);
            PyTuple_SET_ITEM(sig, 4, intent_str);
        }
        Py_XDECREF(fortran_order);
    }
    else {
        if (!settings->ignore_fixed_shape_in_nested_calls) {
            PyObject *rank = PyObject_GetAttr(shape, str_rank);
            PyObject *fortran_order = PyObject_GetAttr(shape, str_fortran_order);
            PyObject *dims = PyObject_GetAttr(shape, str_dims);
            sig = PyTuple_New(6);
            if (sig) {
                Py_INCREF(name);
                PyTuple_SET_ITEM(sig, 0, name);
                PyTuple_SET_ITEM(sig, 1, numpy_dtype);
                PyTuple_SET_ITEM(sig, 2, rank ? rank : Py_None);
                PyTuple_SET_ITEM(sig, 3, fortran_order ? fortran_order : Py_False);
                Py_INCREF(intent_str);
                PyTuple_SET_ITEM(sig, 4, intent_str);
                PyTuple_SET_ITEM(sig, 5, dims ? dims : Py_None);
            }
            Py_XDECREF(fortran_order);
        } else {
            sig = PyTuple_Pack(5, name, numpy_dtype, Py_None, Py_False, intent_str);
        }
    }
    
    Py_DECREF(shape);
    Py_DECREF(dtype);
    return sig;
}

// Main signature extraction - optimized dispatcher
static PyObject* get_signature_from_arg(PyObject *arg, PyObject *name, 
                                         int *to_execute, Settings *settings) {
    // Fast paths in order of likelihood
    if (PyArray_Check(arg)) {
        return sig_from_ndarray(arg, name);
    }
    
    if (PyLong_Check(arg) || PyFloat_Check(arg) || PyComplex_Check(arg)) {
        return sig_from_scalar(arg, name);
    }
    
    if (is_instance_fast(arg, NumpyGeneric)) {
        return sig_from_numpy_scalar(arg, name);
    }
    
    if (is_instance_fast(arg, ArrayType)) {
        return sig_from_arraytype(arg, name, to_execute, settings);
    }
    
    if (is_instance_fast(arg, ExpressionNode)) {
        return sig_from_expression_node(arg, name, to_execute, settings);
    }
    
    // DataType handling
    if (is_instance_fast(arg, DataType) && PyObject_IsSubclass((PyObject*)Py_TYPE(arg), DataType)) {
        if (PyType_Check(arg) && PyObject_IsSubclass(arg, DataType)) {
            *to_execute = 0;
            PyObject *numpy_dtype = PyObject_CallMethodObjArgs(arg, str_get_numpy, NULL);
            if (!numpy_dtype) return NULL;
            PyObject *sig = PyTuple_Pack(2, name, numpy_dtype);
            return sig;
        }
    }
    
    PyErr_Format(PyExc_ValueError, "Argument %R type not supported", name);
    return NULL;
}

// Pre-calculate total signature size to avoid list reallocations
static PyObject* get_signature_and_runtime_args(PyObject *self, PyObject *args) {
    PyObject *py_args;
    PyObject *kwargs;
    PyObject *params;
    PyObject *fixed_param_indices;
    int n_pos_def_args;
    char *catch_var_name_str;
    int add_shape_descriptors;
    int ignore_fixed_shape;
    int reorder_kwargs;

    if (!PyArg_ParseTuple(args, "OOOOisipp", &py_args, &kwargs, &params, &fixed_param_indices, 
                          &n_pos_def_args, &catch_var_name_str, 
                          &add_shape_descriptors, &ignore_fixed_shape, &reorder_kwargs)) {
        return NULL;
    }

    Settings settings = {add_shape_descriptors, ignore_fixed_shape, reorder_kwargs};
    int to_execute = 1;
    
    Py_ssize_t args_len = PyTuple_Size(py_args);
    Py_ssize_t num_fixed = PyList_Size(fixed_param_indices);
    
    // Pre-calculate total signature size
    PyObject *unused_kwargs = PyDict_Copy(kwargs);
    if (!unused_kwargs) return NULL;
    
    // Count variable args and kwargs
    Py_ssize_t pos_idx = 0;
    for (Py_ssize_t i = 0; i < num_fixed; i++) {
        PyObject *param_idx_obj = PyList_GetItem(fixed_param_indices, i);
        Py_ssize_t param_idx = PyLong_AsSsize_t(param_idx_obj);
        PyObject *param = PyList_GetItem(params, param_idx);
        PyObject *p_kind = PyObject_GetAttr(param, str_kind);
        long kind = PyLong_AsLong(p_kind);
        Py_DECREF(p_kind);
        
        if (kind == KIND_POSITIONAL_ONLY || kind == KIND_POSITIONAL_OR_KEYWORD) {
            if (pos_idx < args_len) {
                pos_idx++;
            }
        }
    }
    
    Py_ssize_t var_args_count = args_len - pos_idx;
    Py_ssize_t var_kwargs_count = PyDict_Size(unused_kwargs);
    Py_ssize_t total_sig_size = num_fixed + var_args_count + var_kwargs_count;
    
    // Pre-allocate with exact sizes - no reallocations!
    PyObject *runtime_args = PyList_New(0);  // Will use SET_ITEM after
    PyObject *signature = PyList_New(total_sig_size);
    if (!runtime_args || !signature) {
        Py_XDECREF(runtime_args);
        Py_XDECREF(signature);
        Py_DECREF(unused_kwargs);
        return NULL;
    }
    
    // Reset for actual processing
    pos_idx = 0;
    Py_ssize_t sig_idx = 0;
    
    for (Py_ssize_t i = 0; i < num_fixed; i++) {
        PyObject *param_idx_obj = PyList_GetItem(fixed_param_indices, i);
        Py_ssize_t param_idx = PyLong_AsSsize_t(param_idx_obj);
        PyObject *param = PyList_GetItem(params, param_idx);
        
        PyObject *p_kind = PyObject_GetAttr(param, str_kind);
        PyObject *p_name = PyObject_GetAttr(param, str_name);
        PyObject *p_default = PyObject_GetAttr(param, str_default);
        PyObject *p_is_comptime = PyObject_GetAttr(param, str_is_comptime);
        
        if (!p_kind || !p_name || !p_default || !p_is_comptime) {
            Py_XDECREF(p_kind); Py_XDECREF(p_name); Py_XDECREF(p_default); Py_XDECREF(p_is_comptime);
            Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
            return NULL;
        }

        long kind = PyLong_AsLong(p_kind);
        int is_comptime = PyObject_IsTrue(p_is_comptime);
        PyObject *arg = NULL;
        
        if (kind == KIND_POSITIONAL_ONLY || kind == KIND_POSITIONAL_OR_KEYWORD) {
            if (pos_idx < args_len) {
                arg = PyTuple_GetItem(py_args, pos_idx);
                Py_INCREF(arg);
                pos_idx++;
            } else {
                arg = PyDict_GetItem(unused_kwargs, p_name);
                if (arg) {
                    Py_INCREF(arg);
                    PyDict_DelItem(unused_kwargs, p_name);
                } else if (p_default != INSPECT_EMPTY) {
                    arg = p_default;
                    Py_INCREF(arg);
                } else {
                    PyErr_Format(PyExc_ValueError, "Missing required argument: %S", p_name);
                    Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                    Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
                    return NULL;
                }
            }
        } else {
            arg = PyDict_GetItem(unused_kwargs, p_name);
            if (arg) {
                Py_INCREF(arg);
                PyDict_DelItem(unused_kwargs, p_name);
            } else if (p_default != INSPECT_EMPTY) {
                arg = p_default;
                Py_INCREF(arg);
            } else {
                PyErr_Format(PyExc_ValueError, "Missing required argument: %S", p_name);
                Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
                return NULL;
            }
        }
        
        if (is_comptime) {
            PyList_SET_ITEM(signature, sig_idx++, arg);
        } else {
            PyObject *sig_item = get_signature_from_arg(arg, p_name, &to_execute, &settings);
            if (!sig_item) {
                Py_DECREF(arg);
                Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
                Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
                return NULL;
            }
            PyList_SET_ITEM(signature, sig_idx++, sig_item);
            PyList_Append(runtime_args, arg);
            Py_DECREF(arg);
        }
        
        Py_DECREF(p_kind); Py_DECREF(p_name); Py_DECREF(p_default); Py_DECREF(p_is_comptime);
    }
    
    // Catch *args - use cached name pattern
    if (pos_idx < args_len) {
        for (Py_ssize_t j = 0; pos_idx < args_len; pos_idx++, j++) {
            PyObject *arg = PyTuple_GetItem(py_args, pos_idx);
            PyObject *name_tuple = PyTuple_Pack(2, PyUnicode_FromString(catch_var_name_str), 
                                               PyLong_FromLong(j));
            if (!name_tuple) {
                Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
                return NULL;
            }
            
            PyObject *sig_item = get_signature_from_arg(arg, name_tuple, &to_execute, &settings);
            Py_DECREF(name_tuple);
            if (!sig_item) {
                Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
                return NULL;
            }
            
            PyList_SET_ITEM(signature, sig_idx++, sig_item);
            PyList_Append(runtime_args, arg);
        }
    }
    
    // Catch **kwargs
    PyObject *keys = PyDict_Keys(unused_kwargs);
    if (settings.reorder_kwargs) {
        PyList_Sort(keys);
    }
    
    Py_ssize_t num_keys = PyList_Size(keys);
    for (Py_ssize_t k = 0; k < num_keys; k++) {
        PyObject *key = PyList_GetItem(keys, k);
        PyObject *val = PyDict_GetItem(unused_kwargs, key);
        
        PyObject *sig_item = get_signature_from_arg(val, key, &to_execute, &settings);
        if (!sig_item) {
            Py_DECREF(keys); Py_DECREF(unused_kwargs); Py_DECREF(runtime_args); Py_DECREF(signature);
            return NULL;
        }
        PyList_SET_ITEM(signature, sig_idx++, sig_item);
        PyList_Append(runtime_args, val);
    }
    Py_DECREF(keys);
    Py_DECREF(unused_kwargs);
    
    // Trim signature list to actual size (in case of errors in counting)
    if (sig_idx < total_sig_size) {
        PyList_SetSlice(signature, sig_idx, total_sig_size, NULL);
    }
    
    PyObject *sig_tuple = PyList_AsTuple(signature);
    Py_DECREF(signature);
    
    PyObject *result = PyTuple_Pack(3, to_execute ? Py_True : Py_False, sig_tuple, runtime_args);
    Py_DECREF(sig_tuple);
    Py_DECREF(runtime_args);
    
    return result;
}

static PyObject *init_globals(PyObject *self, PyObject *args) {
    PyObject *types_dict;
    PyObject *constants_dict;

    if (!PyArg_ParseTuple(args, "OO", &types_dict, &constants_dict)) {
        return NULL;
    }

    #define LOAD_TYPE(name) \
        name = PyDict_GetItemString(types_dict, #name); \
        Py_XINCREF(name);

    LOAD_TYPE(ArrayType);
    LOAD_TYPE(DataType);
    LOAD_TYPE(ExpressionNode);
    LOAD_TYPE(Variable);
    LOAD_TYPE(GetAttr);
    LOAD_TYPE(GetItem);
    LOAD_TYPE(SCALAR);
    LOAD_TYPE(UNKNOWN);
    LOAD_TYPE(NumpyGeneric);
    #undef LOAD_TYPE

    PyObject *tmp;
    #define LOAD_CONST(name) \
        tmp = PyDict_GetItemString(constants_dict, #name); \
        if (tmp) name = PyLong_AsLong(tmp);

    LOAD_CONST(KIND_POSITIONAL_ONLY);
    LOAD_CONST(KIND_POSITIONAL_OR_KEYWORD);
    LOAD_CONST(KIND_VAR_POSITIONAL);
    LOAD_CONST(KIND_KEYWORD_ONLY);
    LOAD_CONST(KIND_VAR_KEYWORD);
    #undef LOAD_CONST

    INSPECT_EMPTY = PyDict_GetItemString(constants_dict, "INSPECT_EMPTY");
    Py_XINCREF(INSPECT_EMPTY);

    // Init interned strings
    str_kind = PyUnicode_InternFromString("kind");
    str_name = PyUnicode_InternFromString("name");
    str_default = PyUnicode_InternFromString("default");
    str_is_comptime = PyUnicode_InternFromString("is_comptime");
    str_dtype = PyUnicode_InternFromString("dtype");
    str_names = PyUnicode_InternFromString("names");
    str_shape = PyUnicode_InternFromString("shape");
    str_get_numpy = PyUnicode_InternFromString("get_numpy");
    str_has_comptime_undefined_dims = PyUnicode_InternFromString("has_comptime_undefined_dims");
    str_fortran_order = PyUnicode_InternFromString("fortran_order");
    str_rank = PyUnicode_InternFromString("rank");
    str_dims = PyUnicode_InternFromString("dims");
    str_variable = PyUnicode_InternFromString("variable");
    str_intent = PyUnicode_InternFromString("intent");
    str_in = PyUnicode_InternFromString("in");
    str_inout = PyUnicode_InternFromString("inout");
    str_underscore_shape = PyUnicode_InternFromString("_shape");

    Py_RETURN_NONE;
}

static PyMethodDef SignatureMethods[] = {
    {"get_signature_and_runtime_args", get_signature_and_runtime_args, METH_VARARGS, "Optimize signature parsing"},
    {"init_globals", init_globals, METH_VARARGS, "Initialize globals"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef signaturemodule = {
    PyModuleDef_HEAD_INIT,
    "_signature",
    NULL,
    -1,
    SignatureMethods
};

PyMODINIT_FUNC PyInit__signature(void) {
    import_array();
    return PyModule_Create(&signaturemodule);
}
