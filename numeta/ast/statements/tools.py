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
    from numeta.ast.module import builtins_module

    dependencies = set()
    declarations = {}

    # revese to preserve the order of the entities
    for entity in entities[::-1]:
        if entity.parent is builtins_module:
            # No need to declare
            continue
        if not for_module:
            if entity.parent is None:
                # It is a local variable, so we need to declare
                declarations[entity.name] = entity.get_declaration()
            elif entity.parent is curr_module:
                # It is in the current module
                continue
            else:
                dependencies.add((entity.parent, entity))
        else:
            if entity.parent is None or entity.parent is curr_module:
                declarations[entity.name] = entity.get_declaration()
            else:
                dependencies.add((entity.parent, entity))

    new_declarations = declarations.copy()
    while new_declarations:
        new_entities = []
        for declaration in new_declarations.values():
            for variable in declaration.extract_entities():
                if variable not in new_entities and variable not in entities:
                    new_entities.append(variable)

        entities.extend(new_entities)

        # Now we can add the dependencies or define the local variables
        new_declarations = {}
        for entity in new_entities:
            if entity.parent is builtins_module:
                continue
            if not for_module:
                if entity.parent is None:
                    if entity.name not in declarations:
                        new_declarations[entity.name] = entity.get_declaration()
                elif entity.parent is curr_module:
                    continue
                else:
                    dependencies.add((entity.parent, entity))
            else:
                if entity.parent is None or entity.parent is curr_module:
                    declarations[entity.name] = entity.get_declaration()
                else:
                    dependencies.add((entity.parent, entity))

        declarations.update(new_declarations)

    return dependencies, {k: v for k, v in reversed(list(declarations.items()))}


def divide_variables_and_derived_types(declarations):
    from .variable_declaration import VariableDeclaration
    from .derived_type_declaration import DerivedTypeDeclaration
    from .subroutine_declaration import SubroutineDeclaration

    variable_declarations = {}
    derived_type_declarations = {}
    subroutine_declaration = {}

    for name, declaration in declarations.items():
        if isinstance(declaration, VariableDeclaration):
            variable_declarations[name] = declaration
        elif isinstance(declaration, DerivedTypeDeclaration):
            derived_type_declarations[name] = declaration
        elif isinstance(declaration, SubroutineDeclaration):
            subroutine_declaration[name] = declaration
        else:
            raise NotImplementedError(f"Unknown declaration type: {declaration}")

    return variable_declarations, derived_type_declarations, subroutine_declaration
