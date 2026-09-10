# Numerical calculus with SciPy

## Definite integrals: `scipy.integrate.quad`

`scipy.integrate.quad(func, a, b)` returns a tuple `(result, abserr)`:
the estimated integral value and an estimate of the absolute error. It uses
adaptive quadrature (QUADPACK under the hood), so it's accurate for smooth
functions over finite or infinite bounds (`a`/`b` can be `-np.inf`/`np.inf`).

```python
import numpy as np
from scipy import integrate
result, abserr = integrate.quad(lambda x: np.exp(-x**2), -2, 2)
print(result, abserr)
```

Always report or check `abserr` alongside the result for anything that will
be quoted as a precise answer — a large `abserr` relative to `result` means
the integrand is likely oscillatory or has a singularity `quad` is
struggling with, and you may need `points=` to flag known singularities or
switch to `scipy.integrate.quad_vec`/`nquad` for multi-dimensional cases.

## Double and triple integrals

`scipy.integrate.dblquad(func, a, b, gfun, hfun)` integrates `func(y, x)`
over `x in [a, b]`, `y in [gfun(x), hfun(x)]` — note the argument order is
`(y, x)` for `func`, which trips people up. `scipy.integrate.nquad` handles
arbitrary dimension with a list of `[a, b]` ranges per variable.

## Root finding

`scipy.optimize.brentq(func, a, b)` finds a root of `func` in `[a, b]`
assuming `func(a)` and `func(b)` have opposite signs (bracketing method,
very reliable when a bracket is known). `scipy.optimize.newton(func, x0)`
uses Newton's method from a single starting guess when you don't have a
bracket, optionally with a derivative `fprime=`.

## Numerical differentiation

For a) simple cases, central difference `(f(x+h) - f(x-h)) / (2*h)` with a
small `h` (like `1e-6`) is usually good enough. For anything precision-
sensitive, prefer symbolic differentiation via SymPy (`sympy_symbolic.md`)
and evaluate the resulting expression numerically instead.
