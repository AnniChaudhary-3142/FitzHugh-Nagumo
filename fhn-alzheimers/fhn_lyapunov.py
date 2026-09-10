"""
First Lyapunov coefficient at the single-neuron Hopf point, from
Kuznetsov's projection formula. Negative is supercritical, positive is
subcritical. Unlike the sweeps, this does not depend on grid resolution.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

B_PARAM, EPS, I_EXT = 0.80, 0.06, 0.30


def fixed_point(a):
    def f(v):
        return v - v**3 / 3.0 + I_EXT - (v + a) / B_PARAM
    grid = np.linspace(-3.0, 3.0, 601)
    sc = np.where(np.diff(np.sign(f(grid))))[0]
    v = brentq(f, grid[sc[0]], grid[sc[0] + 1])
    return v, (v + a) / B_PARAM


def a_hopf():
    return brentq(lambda a: fixed_point(a)[0] ** 2 - (1.0 - EPS * B_PARAM), 0.4, 1.2)


def first_lyapunov():
    aH = a_hopf()
    v_star, w_star = fixed_point(aH)

    # Jacobian at the Hopf point
    A = np.array([[1 - v_star**2, -1.0],
                  [EPS,          -EPS * B_PARAM]])
    w0 = np.sqrt(np.linalg.det(A))          # imaginary part of the eigenvalues

    # right eigenvector q: A q = i w0 q ; left eigenvector p: A^T p = -i w0 p
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(np.imag(vals)))       # eigenvalue +i w0
    q = vecs[:, k]
    valsT, vecsT = np.linalg.eig(A.T)
    kt = int(np.argmin(np.imag(valsT)))     # eigenvalue -i w0
    p = vecsT[:, kt]
    p = p / np.conj(np.vdot(p, q))          # normalise so <p,q> = conj(p).q = 1
    # (np.vdot(p,q) already conjugates the first argument)

    # exact multilinear forms (only first component is nonzero)
    def Bf(x, y):
        return np.array([-2.0 * v_star * x[0] * y[0], 0.0], dtype=complex)

    def Cf(x, y, z):
        return np.array([-2.0 * x[0] * y[0] * z[0], 0.0], dtype=complex)

    qb = np.conj(q)
    g20 = np.vdot(p, Bf(q, q))
    g11 = np.vdot(p, Bf(q, qb))
    g21 = np.vdot(p, Cf(q, q, qb))

    l1 = (1.0 / (2.0 * w0)) * np.real(g21) \
         - (1.0 / (2.0 * w0**2)) * np.imag(g20 * g11)

    return dict(a_hopf=float(aH), v_star=float(v_star), omega0=float(w0),
                g20=complex(g20), g11=complex(g11), g21=complex(g21),
                l1=float(l1))


if __name__ == "__main__":
    r = first_lyapunov()
    print(f"a_Hopf   = {r['a_hopf']:.4f}")
    print(f"v*       = {r['v_star']:.4f}   (v*)^2 = {r['v_star']**2:.4f}")
    print(f"omega0   = {r['omega0']:.4f}")
    print(f"g20={r['g20']:.4f}  g11={r['g11']:.4f}  g21={r['g21']:.4f}")
    print(f"first Lyapunov coefficient l1 = {r['l1']:.5f}")
    verdict = "SUPERCRITICAL" if r['l1'] < 0 else "SUBCRITICAL"
    print(f"\n=> l1 {'<' if r['l1']<0 else '>'} 0  =>  {verdict} Hopf")
    out = {k: (str(v) if isinstance(v, complex) else v) for k, v in r.items()}
    out["verdict"] = verdict.lower()
    Path("alzheimers_hopf").mkdir(exist_ok=True)
    json.dump(out, open("alzheimers_hopf/lyapunov.json", "w"), indent=2)
