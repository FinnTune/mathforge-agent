# Linear algebra with NumPy

## Eigenvalues and eigenvectors

`numpy.linalg.eig(a)` returns `(eigenvalues, eigenvectors)` for a square
matrix `a`. `eigenvalues` is a 1-D complex array (real if `a` is symmetric);
`eigenvectors[:, i]` is the eigenvector for `eigenvalues[i]`, normalized to
unit length. For a symmetric/Hermitian matrix, prefer `numpy.linalg.eigh` —
it's faster and guarantees real, sorted eigenvalues.

To find the eigenvalue with the largest magnitude:

```python
import numpy as np
A = np.array([[2, 1], [1, 3]])
values, vectors = np.linalg.eig(A)
dominant = values[np.argmax(np.abs(values))]
print(dominant)
```

For a 2x2 symmetric matrix `[[a, b], [b, d]]`, the eigenvalues also have a
closed form via the characteristic polynomial `λ² - (a+d)λ + (ad - b²) = 0`:
`λ = ((a+d) ± sqrt((a+d)² - 4(ad-b²))) / 2`. Useful for double-checking a
numeric result symbolically (see `sympy_symbolic.md`).

## Solving linear systems

Prefer `numpy.linalg.solve(A, b)` over computing `inv(A) @ b` — it's both
faster and numerically more stable (it doesn't form the explicit inverse).
Raises `LinAlgError` if `A` is singular.

## Matrix norms and condition number

`numpy.linalg.norm(A, ord=2)` gives the spectral norm (largest singular
value). `numpy.linalg.cond(A)` gives the condition number — a large value
(say > 1e10) is a warning sign that `solve`/`eig` results may be numerically
unreliable for that matrix.

## Determinant and rank

`numpy.linalg.det(A)` and `numpy.linalg.matrix_rank(A)`. Determinant of a
near-singular matrix can be a very small nonzero float due to floating-point
error — check `matrix_rank` or condition number instead of comparing `det`
to exactly zero.
