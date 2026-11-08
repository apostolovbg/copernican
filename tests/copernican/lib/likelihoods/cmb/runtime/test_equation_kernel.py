"""Independent scalar comparisons for grouped declared-equation arithmetic."""

import ast
import unittest

import numpy

from copernican.lib.likelihoods.cmb.runtime.equation_kernel import (
    GroupedEquationProgram,
    _EquationBindings,
    _EquationGroup,
    compile_grouped_equation_program,
    evaluate_scalar_equations,
)
from copernican.lib.likelihoods.cmb.runtime.evolution import (
    _compile_equation_program,
)


class GroupedEquationKernelTestCase(unittest.TestCase):
    """Grouping must preserve coordinates, bindings and expression order."""

    def test_affine_state_proof_retains_nonlinear_and_collision_dependencies(
        self,
    ):
        from copernican.lib.likelihoods.cmb.runtime.equation_kernel import (
            is_affine_state_system,
        )

        specs = ((0, "tau", "gain*x + force", None),)
        self.assertTrue(
            is_affine_state_system(specs, ("x",), (), ()),
            "degree must recognize state-linear arithmetic",
        )
        for expression in ("x*x", "sin(x)", "1/x", "x**2", "abs(x)"):
            self.assertFalse(
                is_affine_state_system(
                    ((0, "tau", expression, None),),
                    ("x",),
                    (),
                    (),
                )
            )
        # Forward references and derivative aliases cannot conceal a
        # nonlinear dependency or a state-dependent collision coefficient.
        self.assertFalse(
            is_affine_state_system(
                ((0, "tau", "b", None),),
                ("x",),
                (("b", "a"), ("a", "x*x")),
                (),
            )
        )
        self.assertFalse(
            is_affine_state_system(
                specs,
                ("x",),
                (),
                (("dx", "x"),),
                ("dx",),
            )
        )
        self.assertFalse(
            is_affine_state_system(
                ((0, "a", "x", None),),
                ("x",),
                (),
                (),
            )
        )
        self.assertTrue(
            is_affine_state_system(
                ((0, "tau", "x**1 + exp(rate)*dx", None),),
                ("x",),
                (),
                (("dx", "x"),),
                ("cos(rate)",),
            )
        )

    def test_boundary_and_rebound_rows_do_not_bind_all_interior_states(self):
        specs = tuple(
            (index, "tau", f"gain * x{index} + x{index + 1}", None)
            for index in range(48)
        )
        # x7 is an explicitly rebound value; x48 is a terminal closure.
        names = tuple((f"x{i}", i) for i in range(48) if i != 7)
        program = compile_grouped_equation_program(specs, names)
        independent = _compile_equation_program(specs)
        states = numpy.random.default_rng(11).normal(size=(48, 7))
        self.assertEqual(program.context_names, {"gain", "x7", "x48"})
        for dtype in (numpy.float64, numpy.longdouble):
            context = {f"x{i}": states[i].astype(dtype) for i in range(48)}
            context.update(gain=1.3, x7=states[7].astype(dtype) + dtype(0.25))
            context["x48"] = numpy.linspace(-1, 1, 7).astype(dtype)
            minimal = {name: context[name] for name in program.context_names}
            actual = numpy.zeros_like(states)
            program(minimal, states, actual, {"tau": 0.7})
            for column in range(states.shape[1]):
                scalar_context = {
                    name: (
                        value[column]
                        if isinstance(value, numpy.ndarray)
                        else value
                    )
                    for name, value in context.items()
                }
                expected = numpy.zeros(states.shape[0])
                independent(
                    scalar_context, states[:, column], expected, {"tau": 0.7}
                )
                numpy.testing.assert_array_equal(actual[:, column], expected)

    def test_scalar_grouping_and_reference_preserve_declared_arithmetic(self):
        specs = tuple(
            (index, "tau", f"gain * x{index} - {index+1}.0 * k", None)
            for index in range(16)
        )
        names = tuple((f"x{index}", index) for index in range(16))
        grouped = compile_grouped_equation_program(specs, names)
        reference = _compile_equation_program(specs)
        states = numpy.linspace(-1, 1, 16)
        context = {name: states[index] for name, index in names}
        context.update(gain=1.3, k=0.05)
        actual, expected = numpy.zeros(16), numpy.zeros(16)
        evaluate_scalar_equations(
            grouped, reference, context, states, actual, {"tau": 0.7}
        )
        reference(context, states, expected, {"tau": 0.7})
        self.assertTrue(
            numpy.allclose(actual, expected, rtol=2e-15, atol=1e-15),
            "evaluate_scalar_equations disagreed with the scalar authority",
        )
        self.assertLess(len(grouped.groups), states.size)

    def test_operand_name_and_constant_visitors_preserve_binding_roles(self):
        rewriter = _EquationBindings()
        self.assertEqual(
            rewriter.operand(ast.Constant(7), "literal", 7.0).slice.value, 0
        )
        named = rewriter.visit_Name(ast.Name(id="density", ctx=ast.Load()))
        self.assertEqual(rewriter.bindings[1], ("context", "density"))
        self.assertEqual(named.slice.value, 1)
        constant = rewriter.visit_Constant(ast.Constant(2.5))
        self.assertEqual(rewriter.bindings[2], ("literal", 2.5))
        self.assertEqual(constant.slice.value, 2)

    def test_equation_groups_match_scalar_authority_for_every_mode(self):
        specs = tuple(
            (index, "tau", f"gain * x{index} - {index+1}.0 * k", None)
            for index in range(32)
        ) + (
            (32, "a", "2.0 * x0 - x1 / (1.0 + k**2)", None),
            (33, "z", "sqrt(abs(x2)) + sin(k)", None),
            (34, "tau", None, 0),
        )
        state_names = tuple((f"x{index}", index) for index in range(32))
        program = compile_grouped_equation_program(specs, state_names)
        self.assertIsInstance(program, GroupedEquationProgram)
        self.assertIsInstance(program.groups[0], _EquationGroup)
        self.assertLess(len(program.groups), 6)
        states = numpy.random.default_rng(4).normal(size=(35, 7))
        context = {name: states[index] for name, index in state_names}
        context.update(gain=2.0, k=numpy.linspace(0.01, 0.3, 7))
        rates = {"tau": 1.0, "a": 0.001, "z": -10.0}
        actual = numpy.zeros_like(states)
        program(context, states, actual, rates)
        independent = _compile_equation_program(specs)
        for column in range(7):
            row_context = {
                name: (
                    value[column]
                    if isinstance(value, numpy.ndarray)
                    else value
                )
                for name, value in context.items()
            }
            expected = numpy.zeros(35)
            independent(row_context, states[:, column], expected, rates)
            numpy.testing.assert_allclose(
                actual[:, column], expected, rtol=2e-15, atol=1e-15
            )
        self.assertIs(
            compile_grouped_equation_program(specs, state_names), program
        )

    def test_context_rebinding_and_literal_roles_remain_distinct(self):
        # A closure can replace a context symbol: excluding it from direct
        # state bindings preserves that contract rather than reading raw slots.
        program = compile_grouped_equation_program(
            (
                (0, "tau", "gain * x", None),
                (1, "tau", "2.0 * x", None),
            ),
            (),
        )
        result = numpy.zeros((2, 3))
        program(
            {"gain": 3.0, "x": numpy.full(3, 7.0)},
            numpy.zeros((2, 3)),
            result,
            {"tau": 1.0},
        )
        numpy.testing.assert_array_equal(result, ((21.0,) * 3, (14.0,) * 3))
        self.assertEqual(len(_EquationBindings().bindings), 0)

    def test_unreviewed_expression_nodes_are_rejected(self):
        for expression in ("x.real", "x[0]", "open(x)"):
            with self.assertRaises(ValueError):
                compile_grouped_equation_program(
                    ((0, "tau", expression, None),), ()
                )

    def test_varying_context_pool_matches_scalar_equations(self):
        specs = tuple(
            (index, "tau", f"c{index % 3} * x{index}", None)
            for index in range(12)
        )
        states = numpy.arange(60, dtype=float).reshape(12, 5) / 7
        context = {f"x{index}": states[index] for index in range(12)}
        program = compile_grouped_equation_program(specs, ())
        independent = _compile_equation_program(specs)
        for mixed in (False, True):
            context.update(
                {
                    f"c{index}": (
                        numpy.linspace(1, 2, 5) + index
                        if mixed and index
                        else index + 0.5
                    )
                    for index in range(3)
                }
            )
            actual = numpy.zeros_like(states)
            program(context, states, actual, {"tau": 0.7})
            for column in range(5):
                scalar_context = {
                    name: (
                        value[column]
                        if isinstance(value, numpy.ndarray)
                        else value
                    )
                    for name, value in context.items()
                }
                expected = numpy.zeros(12)
                independent(
                    scalar_context, states[:, column], expected, {"tau": 0.7}
                )
                numpy.testing.assert_allclose(
                    actual[:, column], expected, rtol=2e-15, atol=1e-15
                )
