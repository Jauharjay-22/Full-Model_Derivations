"""
Problem 1: Exponential Decay — Option A
All 3 step sizes in ONE figure using a 3×2 grid.
Each row = one step size; left col = solution curve, right col = absolute error.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── ODE definition ─────────────────────────────────────────────
def f(x, y):
    return -y

def exact(x):
    return np.exp(-x)

# ── RK4 ────────────────────────────────────────────────────────
def rk4(f, x0, y0, h, N):
    xs, ys = [x0], [y0]
    x, y = x0, y0
    for _ in range(N):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y  = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x  = x + h
        xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)

# ── Adams Predictor-Corrector (PECE) ───────────────────────────
def adams_pc(f, x0, y0, h, N):
    xs_s, ys_s = rk4(f, x0, y0, h, 3)
    xs = list(xs_s)
    ys = list(ys_s)
    fs = [f(xs[i], ys[i]) for i in range(4)]
    for n in range(3, N):
        x_new = xs[n] + h
        yp = ys[n] + (h/24)*(55*fs[n] - 59*fs[n-1] + 37*fs[n-2] - 9*fs[n-3])
        fp = f(x_new, yp)
        yc = ys[n] + (h/24)*(9*fp + 19*fs[n] - 5*fs[n-1] + fs[n-2])
        fc = f(x_new, yc)
        xs.append(x_new); ys.append(yc); fs.append(fc)
    return np.array(xs), np.array(ys)

# ── Compute results ─────────────────────────────────────────────
step_sizes = [0.1, 0.01, 0.001]
results = {}
for h in step_sizes:
    N = int(round(1.0 / h))
    x_rk4, y_rk4 = rk4(f, 0.0, 1.0, h, N)
    x_apc, y_apc = adams_pc(f, 0.0, 1.0, h, N)
    results[h] = {
        "x_rk4": x_rk4, "y_rk4": y_rk4,
        "x_apc": x_apc, "y_apc": y_apc,
        "err_rk4": np.abs(exact(x_rk4) - y_rk4),
        "err_apc": np.abs(exact(x_apc) - y_apc),
    }

# ── Option A: 3×2 grid figure ──────────────────────────────────
x_fine = np.linspace(0, 1, 400)

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
fig.suptitle("Problem 1: Exponential Decay  —  $dy/dx = -y$,  $y(0)=1$\n"
             "",
             fontsize=13, fontweight='bold')

for row, h in enumerate(step_sizes):
    r = results[h]
    N = int(round(1.0 / h))
    step = 1 if h == 0.1 else (10 if h == 0.01 else 100)
    idx = list(range(0, N+1, step))

    # Left: solution curves
    ax_sol = axes[row, 0]
    ax_sol.plot(x_fine, exact(x_fine), 'k-', lw=2, label='Exact $e^{-x}$')
    ax_sol.plot(r["x_rk4"][idx], r["y_rk4"][idx], 'b--o', ms=4, label='RK4')
    ax_sol.plot(r["x_apc"][idx], r["y_apc"][idx], 'r:s',  ms=4, label='Adams-PC')
    ax_sol.set_xlabel('x'); ax_sol.set_ylabel('y(x)')
    ax_sol.set_title(f'Solution Curve  (h = {h})')
    ax_sol.legend(fontsize=9); ax_sol.grid(True, alpha=0.3)

    # Right: absolute error
    ax_err = axes[row, 1]
    ax_err.semilogy(r["x_rk4"][idx], r["err_rk4"][idx] + 1e-18,
                    'b-o', ms=4, label='RK4')
    ax_err.semilogy(r["x_apc"][idx], r["err_apc"][idx] + 1e-18,
                    'r--s', ms=4, label='Adams-PC')
    ax_err.set_xlabel('x'); ax_err.set_ylabel('Absolute Error (log scale)')
    ax_err.set_title(f'Absolute Error  (h = {h})')
    ax_err.legend(fontsize=9); ax_err.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("Problem_01.png", dpi=150, bbox_inches='tight')
print("Saved -> option_a_grid.png")
plt.show()
