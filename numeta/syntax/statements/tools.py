def print_block(blocks, indent=0, prefix=""):
    indentation = "    " * indent
    prefix_indentation = prefix + indentation
    prefix_indent_length = len(prefix_indentation)

    lines_to_print = [""]
    lines_lengths = [0]

    for string in blocks:
        current_line_length = lines_lengths[-1]
        new_length = current_line_length + len(string)
        total_length = prefix_indent_length + new_length + len(" &")

        if total_length < 120:
            lines_to_print[-1] += string
            lines_lengths[-1] = new_length
        else:
            lines_to_print[-1] += " &"
            lines_to_print.append(string)
            lines_lengths.append(len(string))

    result = "\n".join(prefix_indentation + line for line in lines_to_print) + "\n"
    return result


def get_shape_blocks(shape, fortran_order=True):
    """
    Construct the code blocks for the shape of an array.
    If fortran_order is False, the shape will be reversed.
    E.g.
        - shape (slice(2, 4, None), 5)

            -> ["(", "2", ":", "4", ",", "5", ")"] # fortran_order=True
            -> ["(", "5", ",", "2", ":", "4", ")"] # fortran_order=False

    """

    from numeta.syntax.syntax_settings import settings

    lbound = settings.array_lower_bound

    result = ["("]

    def convert(element):
        if element is None:
            # Assumed size array
            return [str(lbound), ":", "*"]
        elif isinstance(element, int):
            return [str(lbound), ":", str(element + (lbound - 1))]
        elif isinstance(element, slice):
            shift_end = None
            if element.start is None:
                start = [str(lbound)]

                shift_end = lbound - 1

            elif isinstance(element.start, int):
                start = [str(element.start)]
            else:
                start = element.start.get_code_blocks()

            # Note if the start is None, it has not be setted so we have to shift the end

            if element.stop is None:
                stop = [""]
            elif isinstance(element.stop, int):
                if shift_end is not None:
                    stop = [str(element.stop + shift_end)]
                else:
                    stop = [str(element.stop)]
            else:
                if shift_end is not None:
                    stop = (element.stop + shift_end).get_code_blocks()
                else:
                    stop = element.stop.get_code_blocks()

            if element.step is not None:
                raise NotImplementedError("Step in array dimensions is not implemented yet")
            return start + [":"] + stop

        else:
            # probably a variable or an expression
            return [
                str(lbound),
                ":",
                *(element + (lbound - 1)).get_code_blocks(),
            ]

    if isinstance(shape, tuple):
        dims = [convert(d) for d in shape]

        if not fortran_order:
            dims = dims[::-1]

        result += dims[0]
        for dim in dims[1:]:
            result += [",", " "]
            result += dim
    else:
        result += convert(shape)

    result += [")"]

    return result


# def get_variables(element):
#
#    if hasattr(element, "get_variables"):
#        return element.get_variables()
#
#    elif isinstance(element, (tuple, list)):
#        variables = []
#
#        for e in element:
#            variables += get_variables(e)
#
#        return variables
#
#    elif isinstance(element, slice):
#        variables = []
#        variables += get_variables(element.start)
#        variables += get_variables(element.stop)
#        variables += get_variables(element.step)
#        return variables
#    else:
#        return []


def get_nested_dependencies_or_declarations(entities, curr_module, for_module=False):
    """
    This function takes a set of entities and returns the dependencies and declarations of the entities.
    If for_module is True, it will return the declarations of the entities that are in the same module as curr_module.
    Important: Reverse the order of the declarations to make sure that the dependencies are declared before the entities.
    """
    from numeta.syntax.module import builtins_module

    dependencies = set()
    declarations = {}

    # revese to preserve the order of the entities
    for entity in entities[::-1]:
        if entity.module is builtins_module:
            continue
        if not for_module:
            if entity.module is None:
                declarations[entity.name] = entity.get_declaration()
            elif entity.module is curr_module:
                continue
            else:
                dependencies.add((entity.module, entity))
        else:
            if entity.module is None or entity.module is curr_module:
                declarations[entity.name] = entity.get_declaration()
            else:
                dependencies.add((entity.module, entity))

    new_declarations = declarations.copy()
    while new_declarations:
        new_entities = []
        for declaration in new_declarations.values():
            if hasattr(declaration, "extract_declaration_entities"):
                for variable in declaration.extract_declaration_entities():
                    if variable not in new_entities and variable not in entities:
                        new_entities.append(variable)
            else:
                for variable in declaration.extract_entities():
                    if variable not in new_entities and variable not in entities:
                        new_entities.append(variable)

        entities.extend(new_entities)

        # Now we can add the dependencies or define the local variables
        new_declarations = {}
        for entity in new_entities:
            if entity.module is builtins_module:
                continue
            if not for_module:
                if entity.module is None:
                    if entity.name not in declarations:
                        new_declarations[entity.name] = entity.get_declaration()
                elif entity.module is curr_module:
                    continue
                else:
                    dependencies.add((entity.module, entity))
            else:
                if entity.module is None or entity.module is curr_module:
                    declarations[entity.name] = entity.get_declaration()
                else:
                    dependencies.add((entity.module, entity))

        declarations.update(new_declarations)

    return dependencies, {k: v for k, v in reversed(list(declarations.items()))}


def divide_variables_and_derived_types(declarations):
    from .variable_declaration import VariableDeclaration
    from .derived_type_declaration import DerivedTypeDeclaration

    variable_declarations = {}
    derived_type_declarations = {}

    for name, declaration in declarations.items():
        if isinstance(declaration, VariableDeclaration):
            variable_declarations[name] = declaration
        elif isinstance(declaration, DerivedTypeDeclaration):
            derived_type_declarations[name] = declaration
        else:
            raise NotImplementedError(f"Unknown declaration type: {declaration}")

    return variable_declarations, derived_type_declarations
