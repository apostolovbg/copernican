"""Batch structurally identical declared equations without changing their AST.

Hierarchy equations differ primarily in state indices and numeric factors.
Grouping those bindings adds an equation axis to the existing mode axis;
the arithmetic tree and its evaluation order remain the declared ones.
"""

import ast
import copy
from dataclasses import dataclass
from functools import lru_cache

import numpy

from ....model_adapter import (
    _ALLOWED_CONSTANTS,
    _ALLOWED_MATH_FUNCS,
    _parse_safe_expression,
    _validate_safe_expression,
)


def is_affine_state_system(
    equation_specs,
    state_names,
    value_expressions,
    derivative_bindings,
    collision_expressions=(),
):
    """Prove affine state dependence from the validated arithmetic graph.

    Background values may vary with time. Functions of a state, products of
    states and state-dependent denominators are conservatively nonlinear.
    This classification never infers linearity from sampled physical data.
    """

    degrees = dict.fromkeys(state_names, 1)

    def degree(node):
        """Return polynomial degree, saturated at unsupported/nonlinear."""

        if isinstance(node, ast.Expression):
            return degree(node.body)
        if isinstance(node, ast.Constant):
            return 0
        if isinstance(node, ast.Name):
            return degrees.get(node.id, 0)
        if isinstance(node, ast.UnaryOp):
            return degree(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = degree(node.left), degree(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(left, right)
            if isinstance(node.op, ast.Mult):
                return min(left + right, 2)
            if isinstance(node.op, ast.Div):
                return left if right == 0 else 2
            if isinstance(node.op, ast.Pow):
                if left == right == 0:
                    return 0
                if (
                    isinstance(node.right, ast.Constant)
                    and node.right.value == 1
                ):
                    return left
            return 2
        if isinstance(node, ast.Call):
            return 0 if all(degree(arg) == 0 for arg in node.args) else 2
        return 2

    for output, slot in derivative_bindings:
        degrees[output] = degrees.get(slot, 1)
    expressions = tuple(
        (name, _parse_safe_expression(str(expression)))
        for name, expression in value_expressions
    )
    changed = True
    while changed:
        changed = False
        for name, expression in expressions:
            result = max(degrees.get(name, 0), degree(expression))
            if result != degrees.get(name, 0):
                degrees[name] = result
                changed = True
    for _, coordinate, expression, _ in equation_specs:
        if coordinate not in {"eta", "tau"}:
            return False
        if (
            expression is not None
            and degree(_parse_safe_expression(str(expression))) > 1
        ):
            return False
    return all(
        degree(_parse_safe_expression(str(expression))) == 0
        for expression in collision_expressions
    )


class _EquationBindings(ast.NodeTransformer):
    """Replace runtime symbols and literal factors by ordered operand slots."""

    def __init__(self):
        """Start an independent operand table for one arithmetic tree."""

        self.bindings = []

    def operand(self, node, kind, value):
        """Bind one occurrence without simplifying or reassociating it."""

        index = len(self.bindings)
        self.bindings.append((kind, value))
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id="operands", ctx=ast.Load()),
                slice=ast.Constant(index),
                ctx=ast.Load(),
            ),
            node,
        )

    def visit_Name(self, node):
        """Preserve reviewed functions/constants and bind declared symbols."""

        if node.id in _ALLOWED_MATH_FUNCS or node.id in _ALLOWED_CONSTANTS:
            return node
        return self.operand(node, "context", node.id)

    def visit_Constant(self, node):
        """Keep numeric factors as float64 equation-column inputs."""

        return self.operand(node, "literal", float(node.value))


@dataclass(frozen=True)
class _EquationGroup:
    """One arithmetic program and its equation-specific runtime bindings."""

    targets: numpy.ndarray
    wrt: str
    evaluate: object
    operands: tuple


@dataclass(frozen=True)
class GroupedEquationProgram:
    """Execute all declared state derivatives with one call per AST group."""

    groups: tuple[_EquationGroup, ...]
    promotions: tuple
    context_names: frozenset[str]

    def __call__(self, context, states, derivative, coordinate_rates):
        """Bind real source/state arrays, preserving declared coordinates."""

        mode_count = states.shape[1]
        for target, source, wrt in self.promotions:
            derivative[target] = states[source] * coordinate_rates[wrt]
        for group in self.groups:
            operands = []
            for kind, values in group.operands:
                if kind == "literal":
                    value = values
                elif kind == "state":
                    value = states[values]
                elif kind == "shared":
                    value = context[values]
                elif kind == "mixed":
                    indices, replacements = values
                    resolved = [
                        (rows, numpy.asarray(context[name]))
                        for name, rows in replacements
                    ]
                    dtype = numpy.result_type(
                        states.dtype, *(item.dtype for _, item in resolved)
                    )
                    value = states[indices].astype(dtype, copy=False)
                    for rows, replacement in resolved:
                        value[rows] = numpy.broadcast_to(
                            replacement, (mode_count,)
                        )
                else:
                    names, indices = values
                    resolved = [numpy.asarray(context[name]) for name in names]
                    if all(item.ndim == 0 for item in resolved):
                        pool = numpy.asarray(resolved)[:, None]
                    else:
                        pool = numpy.stack(
                            [
                                numpy.broadcast_to(item, (mode_count,))
                                for item in resolved
                            ]
                        )
                    value = pool[indices]
                operands.append(value)
            values = group.evaluate(operands) * coordinate_rates[group.wrt]
            derivative[group.targets] = values


def evaluate_scalar_equations(
    grouped, reference, context, states, derivative, rates
):
    """Use profitable arithmetic grouping, retaining the scalar authority.

    Both executors receive the same declared operands and coordinates.
    The independent reference remains callable for numerical comparisons.
    """

    if states.ndim == 2:
        grouped(context, states, derivative, rates)
    elif len(grouped.groups) < states.size:
        grouped(context, states[:, None], derivative[:, None], rates)
    else:
        reference(context, states, derivative, rates)


@lru_cache(maxsize=128)
def compile_grouped_equation_program(slot_specs, state_names):
    """Compile validated arithmetic once for each shared hierarchy stencil."""

    state_indices = dict(state_names)
    collected = {}
    promotions = []
    for index, wrt, expression, promotion in slot_specs:
        if expression is None:
            promotions.append((int(index), int(promotion), str(wrt)))
            continue
        parsed = copy.deepcopy(_parse_safe_expression(expression))
        names = {
            node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)
        }
        _validate_safe_expression(expression, names)
        rewriter = _EquationBindings()
        transformed = rewriter.visit(parsed)
        signature = (
            str(wrt),
            ast.dump(transformed, include_attributes=False),
            tuple(kind for kind, _ in rewriter.bindings),
        )
        if signature not in collected:
            collected[signature] = (transformed, [])
        collected[signature][1].append((int(index), tuple(rewriter.bindings)))
    groups = []
    context_names = set()
    for (wrt, _, _), (expression, rows) in collected.items():
        evaluator = ast.FunctionDef(
            name="evaluate",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="operands")],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[ast.Return(expression.body)],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(
                body=[evaluator],
                type_ignores=[],
            )
        )
        namespace = {
            "__builtins__": {},
            **_ALLOWED_CONSTANTS,
            **_ALLOWED_MATH_FUNCS,
        }
        # security-scanner: allow validated arithmetic AST compilation.
        exec(  # nosec B102 - the original expression passed the AST validator.
            compile(module, "<declared-cmb-equation-group>", "exec"),
            namespace,
        )
        operands = []
        for position in range(len(rows[0][1])):
            entries = tuple(bindings[position] for _, bindings in rows)
            if entries[0][0] == "literal":
                values = numpy.asarray([value for _, value in entries])[
                    :, None
                ]
                values.flags.writeable = False
                operands.append(("literal", values))
            else:
                names = tuple(value for _, value in entries)
                if len(set(names)) == 1:
                    operands.append(("shared", names[0]))
                    context_names.add(names[0])
                elif all(name in state_indices for name in names):
                    indices = numpy.asarray(
                        [state_indices[name] for name in names]
                    )
                    indices.flags.writeable = False
                    operands.append(("state", indices))
                elif any(name in state_indices for name in names):
                    indices = numpy.asarray(
                        [state_indices.get(name, 0) for name in names]
                    )
                    indices.flags.writeable = False
                    replacements = {}
                    for row, name in enumerate(names):
                        if name not in state_indices:
                            replacements.setdefault(name, []).append(row)
                    replacement_indices = []
                    for name, replacement_rows in replacements.items():
                        row_indices = numpy.asarray(
                            replacement_rows, dtype=int
                        )
                        row_indices.flags.writeable = False
                        replacement_indices.append((name, row_indices))
                    operands.append(
                        (
                            "mixed",
                            (indices, tuple(replacement_indices)),
                        )
                    )
                    context_names.update(replacements)
                else:
                    unique_names = tuple(dict.fromkeys(names))
                    positions = {
                        name: index for index, name in enumerate(unique_names)
                    }
                    indices = numpy.asarray(
                        [positions[name] for name in names]
                    )
                    indices.flags.writeable = False
                    operands.append(("context", (unique_names, indices)))
                    context_names.update(unique_names)
        targets = numpy.asarray([index for index, _ in rows])
        targets.flags.writeable = False
        groups.append(
            _EquationGroup(
                targets, wrt, namespace["evaluate"], tuple(operands)
            )
        )
    return GroupedEquationProgram(
        tuple(groups), tuple(promotions), frozenset(context_names)
    )
