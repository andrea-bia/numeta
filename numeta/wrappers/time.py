from numeta.fortran.external_modules.omp import omp


def time():
    return omp.omp_get_wtime()
