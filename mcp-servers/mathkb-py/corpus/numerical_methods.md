# General numerical & coding patterns

## Fibonacci and simple recurrences

A closed-form loop beats naive recursion for anything beyond toy sizes —
naive recursive Fibonacci is exponential time (repeated subproblems), while
an iterative version is linear:

```python
def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result
```

For very large `n` or when you need it fast and don't need the whole
sequence, matrix exponentiation (`[[1,1],[1,0]]^n`) or `sympy.fibonacci(n)`
(exact, arbitrary precision) are better than either recursive or iterative
Python loops.

## Floating-point comparisons

Never compare floats with `==`. Use `math.isclose(a, b, rel_tol=1e-9)` or
`numpy.isclose`/`numpy.allclose` for arrays. This matters especially after a
chain of numeric operations (integration, matrix solves) where small
rounding error accumulates.

## Vectorize before you loop

A Python `for` loop over a NumPy array is almost always slower than the
vectorized equivalent. Prefer `np.where`, boolean masking, and array
arithmetic over element-by-element loops:

```python
import numpy as np
x = np.arange(1000)
# Slow: [x_i**2 for x_i in x if x_i % 2 == 0]
# Fast:
evens = x[x % 2 == 0]
squares = evens ** 2
```

## Random seeds for reproducibility

When a task involves randomness (Monte Carlo estimates, random sampling),
set a seed (`numpy.random.default_rng(seed=0)`, not the legacy
`numpy.random.seed`) so results are reproducible if asked to verify or
re-run.
