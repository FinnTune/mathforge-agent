# Plotting with Matplotlib in a headless sandbox

## The Agg backend

The sandbox has no display, so it selects the non-interactive `Agg` backend
before `matplotlib.pyplot` is imported (`matplotlib.use("Agg")`). This means
`plt.show()` does nothing useful — always save to a file with `plt.savefig(...)`
instead, then describe the figure in words for the user.

## Saving a figure

```python
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 2 * np.pi, 400)
plt.plot(x, np.sin(x), label="sin(x)")
plt.plot(x, np.cos(x), label="cos(x)")
plt.legend()
plt.savefig("plots/trig.png")
```

Paths are relative to the sandbox workspace; the `plots/` directory already
exists (created by the sandbox preamble) — save there so the file is easy to
find afterward. Always `plt.close()` (or rely on the sandbox process exiting)
after saving in a loop that makes multiple figures, otherwise Matplotlib
keeps every `Figure` object alive in memory.

## Multiple subplots

`fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))` returns a `Figure`
and either a single `Axes` or an array of them (index with `axes[i]` for 1-D,
`axes[i, j]` for 2-D). Call `.plot()`, `.set_title()`, `.set_xlabel()` etc.
on the `Axes` object rather than the `plt` module directly when working with
subplots — it's less ambiguous about which axes you're drawing on.

## Common gotchas

- `plt.savefig` must come *before* `plt.show()`/before the figure is closed —
  in the sandbox, just make sure `savefig` runs at all (no `show()` needed).
- High-resolution output: pass `dpi=150` or higher to `savefig`.
- For scatter plots with many points, `plt.scatter(x, y, s=..., alpha=0.5)` —
  lowering `alpha` helps visualize density/overlap.
