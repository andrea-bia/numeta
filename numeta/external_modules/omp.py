from numeta.syntax import ExternalModule
from numeta.settings import settings


class OmpModule(ExternalModule):
    def __init__(self):
        super().__init__("omp_lib")
        self.add_method("omp_get_thread_num", arguments=[], result_=settings.DEFAULT_INTEGER)
        self.add_method("omp_get_max_threads", arguments=[], result_=settings.DEFAULT_INTEGER)

    def do(self, *args, **kwargs):
        from numeta.syntax import Do, Comment
        from numeta.syntax.statements.tools import print_block

        class OmpComment(Comment):
            """
            Hack to handle OpenMP statements
            """

            def print_lines(self, indent=0):
                return [print_block(self.comment, indent=indent, prefix="!$omp ")]

        class OmpDo(Do):
            def __init__(
                self,
                *args,
                default="private",
                schedule="dynamic",
                private=None,
                shared=None,
                **kwargs,
            ):
                omp_code = ["parallel do", " "]
                omp_code += [f"default({default})", " "]
                omp_code += [f"schedule({schedule})", " "]
                if private is not None:
                    # TODO: Should we add automatically variables declared inside the loop?
                    omp_code += ["private("]
                    for var in private:
                        omp_code += var.get_code_blocks()
                        omp_code += [", "]
                    omp_code[-1] = ")"
                if shared is not None:
                    # TODO: Should we automatically array shapes if explicitly declared as variables?
                    omp_code += ["shared("]
                    for var in shared:
                        omp_code += var.get_code_blocks()
                        omp_code += [", "]
                    omp_code[-1] = ")"
                OmpComment(omp_code, add_to_scope=True)
                super().__init__(*args, **kwargs)

            def __exit__(self, exc_type, exc_value, traceback):
                super().__exit__(exc_type, exc_value, traceback)
                OmpComment("end parallel do", add_to_scope=True)

        return OmpDo(*args, **kwargs)

    def Do(self, *args, **kwargs):
        return self.do(*args, **kwargs)

    def DO(self, *args, **kwargs):
        return self.do(*args, **kwargs)

    def atomic_update_op(self, variable, to_assign, op):
        from numeta.syntax.statements import Assignment, Comment
        from numeta.syntax.expressions.binary_operation_node import (
            BinaryOperationNodeNoPar,
        )

        Comment(f"$omp atomic update", add_to_scope=True)
        Assignment(
            variable,
            BinaryOperationNodeNoPar(variable, op, to_assign),
            add_to_scope=True,
        )

    def atomic_update_add(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def AtomicUpdateAdd(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def ATOMIC_UPDATE_ADD(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "+")

    def atomic_update_sub(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def AtomicUpdateSub(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def ATOMIC_UPDATE_SUB(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "-")

    def atomic_update_mul(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def AtomicUpdateMul(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def ATOMIC_UPDATE_MUL(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "*")

    def atomic_update_div(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")

    def AtomicUpdateDiv(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")

    def ATOMIC_UPDATE_DIV(self, variable, to_assign):
        self.atomic_update_op(variable, to_assign, "/")


omp = OmpModule()
