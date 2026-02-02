# import numpy as np
# import numeta as nm
#
#
# def test_numpy_mem_jit(backend):
#
#    nm.settings.set_default_from_datatype(nm.size_t, iso_c=True)
#
#    @nm.jit(backend=backend)
#    def alloc(a):
#        from numeta.syntax import Variable
#        from numeta.syntax.expressions import ArrayConstructor
#
#        ptr = Variable("ptr", nm.external_modules.iso_c_binding.FPointer_c)
#        nm.numpy_mem.numpy_allocate(ptr, 10000)
#        b = Variable(
#            "b", nm.external_modules.iso_c_binding.FReal64_c, pointer=True, shape=(None, None)
#        )
#        # nm.iso_c.c_f_pointer(ptr, b, ArrayConstructor(10, 10))
#        nm.iso_c.c_f_pointer(ptr, b, ArrayConstructor(10, 10))
#        b[:] = 2.0
#        a[:] = b[:]
#        nm.numpy_mem.numpy_deallocate(ptr)
#        # a[:] = 2.0
#
#    a = np.zeros((10, 10), dtype=np.float64)
#    alloc(a)
#
#    np.testing.assert_array_equal(a, 2.0 * np.ones((10, 10), dtype=np.float64))
#
#    print(alloc.get_symbolic_functions()[0].get_code())
#
#    # sub = alloc.get_symbolic_function(tuple())
#    # lines = sub.print_lines()
#    # assert lines[0] == "subroutine alloc_0() bind(C)\n"
#    # assert set(lines[1:4]) == {
#    #    "    use numpy, only: PyDataMem_NEW\n",
#    #    "    use numpy, only: PyDataMem_FREE\n",
#    #    "    use iso_c_binding, only: c_size_t\n",
#    # }
#    # expected_rest = [
#    #    "    implicit none\n",
#    #    "    integer(c_size_t) :: fc_s1\n",
#    #    "    fc_s1=PyDataMem_NEW(10_c_size_t)\n",
#    #    "    call PyDataMem_FREE(fc_s1)\n",
#    #    "end subroutine alloc_0\n",
#    # ]
#    # assert lines[4:] == expected_rest
#
