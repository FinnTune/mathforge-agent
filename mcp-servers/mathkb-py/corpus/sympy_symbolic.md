# Symbolic math with SymPy

## Declaring symbols

`x, y = sympy.symbols("x y")` creates symbolic variables. Add assumptions
when they matter for simplification, e.g. `sympy.symbols("x", positive=True)`
or `real=True` — SymPy will otherwise keep expressions in a more general
(and sometimes uglier) form because it can't assume a value's sign or
domain.

## Expanding and collecting terms

```python
import sympy as sp
x, y = sp.symbols("x y")
expr = sp.expand((x + y)**4)
collected = sp.collect(expr, x)
```

`sp.expand` multiplies out products/powers; `sp.factor` does the reverse.
`sp.simplify` tries multiple strategies and picks the "simplest" result —
useful interactively but slower and less predictable than a targeted
function like `sp.trigsimp`, `sp.powsimp`, or `sp.radsimp` when you know
what kind of expression you're simplifying.

## Solving equations

`sp.solve(sp.Eq(lhs, rhs), x)` or `sp.solve(expr, x)` (implicitly `expr == 0`)
returns a list of solutions. For a system, pass a list of equations and a
list of symbols: `sp.solve([eq1, eq2], [x, y])`. `sp.solve` is for
closed-form/exact solutions; for equations without one, use
`sp.nsolve(expr, x, x0)` for a numeric root near a starting guess `x0`.

## Integration and differentiation

`sp.integrate(expr, x)` for indefinite, `sp.integrate(expr, (x, a, b))` for
definite integrals — exact when SymPy recognizes a closed form, otherwise it
returns an unevaluated `Integral` object (check with `.has(sp.Integral)`,
or fall back to numeric integration via `scipy.integrate.quad` instead).
`sp.diff(expr, x)` differentiates; `sp.diff(expr, x, 2)` for the second
derivative.

## Going from symbolic to numeric

`expr.subs(x, 3)` substitutes a value; `.evalf()` forces a numeric
(arbitrary-precision) evaluation, e.g. `expr.subs(x, 3).evalf(10)` for 10
significant digits. `sp.lambdify((x, y), expr, "numpy")` compiles a SymPy
expression into a fast NumPy-vectorized function — use this instead of
repeatedly calling `.subs()` in a loop or over an array.
