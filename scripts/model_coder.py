"""Model coder that turns validated JSON into callable Python functions."""
# DEV NOTE (v1.5a): Converts sympy expressions defined in JSON into lambdified functions.

import sympy as sp


def generate_callables(model_data):
    """Generate a dictionary of callables from model equations."""
    z = sp.symbols('z')
    param_syms = [sp.symbols(p['python_var']) for p in model_data['parameters']]
    local_dict = {p['python_var']: sym for p, sym in zip(model_data['parameters'], param_syms)}
    local_dict['z'] = z
    funcs = {}
    for name, expr in model_data.get('equations', {}).items():
        try:
            sym_expr = sp.sympify(expr, locals=local_dict)
            funcs[name] = sp.lambdify((z, *param_syms), sym_expr, 'numpy')
        except Exception as e:
            raise ValueError(f"Failed to parse equation '{name}': {e}")
    return funcs
