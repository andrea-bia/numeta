def print_block(blocks, indent=0, prefix=""):
    indentation = "    " * indent
    prefix_indentation = prefix + indentation
    max_line_len = 120 - len(prefix_indentation) - 2

    lines_to_print = []
    current_line = ""
    current_len = 0

    for string in blocks:
        string_len = len(string)
        if current_len + string_len < max_line_len:
            current_line += string
            current_len += string_len
        else:
            lines_to_print.append(current_line + " &")
            current_line = string
            current_len = string_len

    lines_to_print.append(current_line)
    return "\n".join(prefix_indentation + line for line in lines_to_print) + "\n"


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


def get_nested_dependencies_or_declarations(entities_by_name, curr_namespace, for_namespace=False):
    """
    This function takes a set of entities and returns the dependencies and declarations of the entities.
    If for_namespace is True, it will return the declarations of the entities that are in the same namespace as curr_namespace.
    Important: Reverse the order of the declarations to make sure that the dependencies are declared before the entities.
    """

    from numeta.ast.namespace import builtins_namespace

    if isinstance(entities_by_name, dict):
        entities_by_name = dict(entities_by_name)
    else:
        entities_by_name = {entity.name: entity for entity in entities_by_name}

    dependencies = set()
    declarations = {}

    # reverse to preserve the input order semantics
    for entity in reversed(list(entities_by_name.values())):
        if entity.parent is builtins_namespace:
            # No need to declare
            continue
        if not for_namespace:
            if entity.parent is None:
                # It is a local variable, so we need to declare
                declarations[entity.name] = entity.get_declaration()
            elif entity.parent is curr_namespace:
                # It is in the current namespace
                continue
            else:
                dependencies.add((entity.parent, entity))
        else:
            if entity.parent is None or entity.parent is curr_namespace:
                declarations[entity.name] = entity.get_declaration()
            else:
                dependencies.add((entity.parent, entity))

    new_declarations = declarations.copy()
    while new_declarations:
        new_entities = {}
        for declaration in new_declarations.values():
            for variable in declaration.extract_entities():
                variable_name = variable.name
                if variable_name not in new_entities and variable_name not in entities_by_name:
                    new_entities[variable_name] = variable

        entities_by_name.update(new_entities)

        # Now we can add the dependencies or define the local variables
        new_declarations = {}
        for entity in new_entities.values():
            if entity.parent is builtins_namespace:
                continue
            if not for_namespace:
                if entity.parent is None:
                    if entity.name not in declarations:
                        new_declarations[entity.name] = entity.get_declaration()
                elif entity.parent is curr_namespace:
                    continue
                else:
                    dependencies.add((entity.parent, entity))
            else:
                if entity.parent is None or entity.parent is curr_namespace:
                    declarations[entity.name] = entity.get_declaration()
                else:
                    dependencies.add((entity.parent, entity))

        declarations.update(new_declarations)

    return dependencies, {k: v for k, v in reversed(list(declarations.items()))}


def divide_variables_and_struct_types(declarations):
    from .variable_declaration import VariableDeclaration
    from .struct_type_declaration import StructTypeDeclaration
    from .procedure_declaration import ProcedureDeclaration

    variable_declarations = {}
    struct_type_declarations = {}
    procedure_declaration = {}

    for name, declaration in declarations.items():
        if isinstance(declaration, VariableDeclaration):
            variable_declarations[name] = declaration
        elif isinstance(declaration, StructTypeDeclaration):
            struct_type_declarations[name] = declaration
        elif isinstance(declaration, ProcedureDeclaration):
            procedure_declaration[name] = declaration
        else:
            raise NotImplementedError(f"Unknown declaration type: {declaration}")

    return variable_declarations, struct_type_declarations, procedure_declaration
