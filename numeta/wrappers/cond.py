import ast
import inspect
from numeta.syntax import IF, ELSE, ELSEIF, Scope


class CondHelper:
    curr = None

    @classmethod
    def get_instance(cls):
        if cls.curr is None:
            raise SyntaxError("No if created yet.")
        return cls.curr

    @classmethod
    def set_instance(cls, instance):
        cls.curr = instance

    def __init__(self):
        self.number_if = 0
        self.if_stack = []
        self.source_cache = {}

    def add_if(self, lineno):
        self.if_stack.append(lineno)
        self.number_if += 1

    def pop_if(self):
        return self.if_stack.pop()

    def get_source_cache(self, frame):
        if frame.f_code.co_filename.startswith("<fc_if_"):
            if_id = int(frame.f_code.co_filename.split("_")[-1].split(">")[0])
            source_lines, tree, starting_line_number = self.source_cache[if_id]
            endif_line_number = frame.f_lineno
            if_line_number = self.if_stack.pop()
        else:
            if_id = self.number_if
            self.number_if += 1
            source_lines, starting_line_number = inspect.getsourcelines(frame.f_code)
            # HACK: add a try except block to take care of the indentation
            source_lines = (
                ["try:\n"]
                + source_lines
                + ["except:\n    raise Warning('impossible to parse the code')"]
            )
            tree = ast.parse("".join(source_lines))
            self.source_cache[if_id] = (source_lines, tree, starting_line_number + 1)
            endif_line_number = frame.f_lineno - starting_line_number + 2
            if_line_number = self.if_stack.pop() - starting_line_number + 2

        return source_lines, tree, if_id, if_line_number, endif_line_number

    def analyze_frame_and_get_if(self, frame):
        """
        Analyze the frame, sets the internal variables
        and returns:
            - True if it is an if statement
            - False if it is an elif statement
        """
        if not frame.f_code.co_filename.startswith("<fc_if_"):
            # it is surely an if statement bc it is not called from the ast exec
            # All the elif statements are called from the ast exec
            self.if_stack.append(frame.f_lineno)
            return True

        else:
            # called from the ast exec so it can be an elif statement
            if_id = int(frame.f_code.co_filename.split("_")[-1].split(">")[0])
            source_lines, _, _ = self.source_cache[if_id]

            if source_lines[frame.f_lineno - 1].strip().startswith("elif"):
                return False
            else:
                self.if_stack.append(frame.f_lineno)
                return True

    def finalize(self):
        if len(self.if_stack) > 0:
            raise SyntaxError("Unmatched if statement")


_helper = CondHelper()
CondHelper.set_instance(_helper)


def cond(cond):
    frame = inspect.currentframe().f_back
    helper = CondHelper.get_instance()
    is_if = helper.analyze_frame_and_get_if(frame)

    if is_if:
        IF(cond)
    else:
        ELSEIF(cond)

    return True


def endif():
    frame = inspect.currentframe().f_back

    helper = CondHelper.get_instance()

    (
        source_lines,
        tree,
        if_id,
        if_line_number,
        endif_line_number,
    ) = helper.get_source_cache(frame)

    previous_if = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.lineno == if_line_number:
            previous_if = node
            break

    from collections import deque

    def get_previous_if(node):
        # BFS to find the previous if statement in the tree wrt node

        queue = deque([node])
        while queue:
            current_node = queue.popleft()
            for child in ast.iter_child_nodes(current_node):
                if isinstance(child, ast.Expr) and child.lineno == endif_line_number:
                    return queue[-1]
                queue.append(child)

    if get_previous_if(tree.body[0]).lineno != previous_if.lineno:
        msg = "Unmatched if statement for an endif()\n"
        msg += "Check this if statement\n"
        msg += ast.unparse(previous_if)
        raise SyntaxError(msg)

    write_elif(previous_if, source_lines, frame, if_id)

    # Important: close the scope of the if statement
    Scope.end()


def write_elif(current_if, source_lines, frame, if_id):
    while hasattr(current_if, "orelse") and len(current_if.orelse) != 0:
        else_body = current_if.orelse

        if len(else_body) != 1 or not isinstance(else_body[0], ast.If):
            # it is surely an else statement
            treat_else(else_body, frame, if_id)

            break
        else:
            """
            If the length of the orelse is 1 and it contains an if statement,
            it is probably an elif statement.
            But it could also be an else statement like this:

            else:
                if True:
                    print('hey')

            In this case, the orelse will contain an if statement.
            So we have to check the source code to see if it is an elif or else statement
            """
            current_if = else_body[0]

            if source_lines[current_if.lineno - 1].strip().startswith("elif"):
                # It is an elif statement

                # close the previous scope
                Scope.end()

                # execute the condition and the body
                compiled_cond = compile(
                    ast.Module([current_if]), filename=f"<fc_if_{if_id}>", mode="exec"
                )
                exec(compiled_cond, frame.f_globals, frame.f_locals)

                # to update the locals
                import ctypes

                ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))

            else:
                treat_else(else_body, frame, if_id)
                # it is an else statement
                break


def treat_else(else_body, frame, if_id):
    # close the previous scope
    Scope.end()

    ELSE()

    compiled_if_body = compile(ast.Module(else_body), filename=f"<fc_if_{if_id}>", mode="exec")
    exec(compiled_if_body, frame.f_globals, frame.f_locals)

    # to update the locals
    import ctypes

    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
