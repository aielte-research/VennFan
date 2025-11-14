import numpy as np
from scipy.optimize import fminbound

# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------

def get_sine_curve(
    X,
    i: float,
    N: int,
    p: float = 0.33,
    lmbd: float = 0.75,
    linear: bool = True,
    corrected: bool = False,
    q=0,
):
    """
    Nonlinear sine-like curve for class boundary (π-shifted).

    p : exponent on |sin|
    lmbd : amplitude decay factor in log-frequency
    """
    X = np.asarray(X, float)
    
    if i == N - 1:
        return np.zeros_like(X)
    
    base = np.sin(2**i * X - np.pi)
    if linear:
        amp = (N-i-1) / N
    else:
        amp = lmbd ** (i+q)
    
    sgn = np.sign(base)
    curve = amp * sgn * np.abs(base) ** p
    if i<1 and corrected:
        return np.where(sgn<0, curve, 1)
    return curve

def get_cosine_curve(
    X,
    i: float,
    N: int,
    p: float = 0.33,
    lmbd: float = 0.75,
    linear: bool = True,
    corrected: bool = False,
    q=0,
):
    """
    Nonlinear cosine-like curve for class boundary (π-shifted).

    p : exponent on |sin|
    lmbd : amplitude decay factor in log-frequency
    """
    X = np.asarray(X, float)
    
    if i == N - 1:
        return np.zeros_like(X)

    base = np.cos(2**i/2 * (X+2*np.pi))
    if linear:
        amp = (N-i-1) / N
    else:
        amp = lmbd ** (i+q)
        
    sgn = np.sign(base)
    curve = amp * sgn * np.abs(base) ** p
    if i<2 and corrected:
        return np.where(sgn<0, curve, 1)
    return curve

def vennfan_find_extrema(
    curve_mode: str,
    p: float,
    lmbd: float,
    q: float = 0.01,
    N: int = 6,
    linear: bool = False
) -> tuple[float, float, float, float]:
    """
    Find (x_min, x_max, y_min, y_max) for the vennfan disc mapping by optimizing over X ∈ [0, 2π],
    using the projected class boundary curves as the objective (no hand-derived formulas).

    Speed heuristics:
      - If curve_mode == "sine":
          x_max, y_max, x_min from i=0; y_min from i=1.
      - If curve_mode == "cosine":
          x_max, y_max from i=0; x_min from i=1; y_min from i=2.

    The underlying half-plane boundary is y_old(X) = get_*_curve(X, i, N, p, lmbd_eff(i), linear=False),
    then projected to the disc via ρ(X) = 1 - y_old(X), and
      u(X) = ρ(X) * cos(X),  v(X) = ρ(X) * sin(X).
    We then optimize u(X) and v(X) over X ∈ [0, 2π] to get the extrema.

    Parameters
    ----------
    curve_mode : {"sine","cosine"}
    p          : float > 0, exponent on |sin| / |cos|
    lmbd       : float > 0, base amplitude-decay factor
    q          : float,   initial exponent (amp ~ lmbd**(i+q))
    N          : int,     number of classes (only i indices matter here)
    linear     : bool,    pass-through to get_*_curve; for this use-case, typically False

    Returns
    -------
    (x_min, x_max, y_min, y_max) : tuple of floats
    """

    if p <= 0:
        raise ValueError("p must be > 0.")
    if lmbd <= 0:
        raise ValueError("lmbd must be > 0.")
    if curve_mode not in ("sine", "cosine"):
        raise ValueError("curve_mode must be 'sine' or 'cosine'.")

    # Your helpers use lmbd**(i+0.1). To realize amp = (lmbd)**(i+q) for a specific i,
    # call them with an i-specific effective lambda:
    #     lmbd_eff(i) ** (i + 0.1) = lmbd ** (i + q)  ⇒  lmbd_eff(i) = lmbd ** ((i+q)/(i+0.1))
    def _lmbd_eff_for_i(i: int) -> float:
        base_q = 0.1
        return float(lmbd) ** ((i + float(q)) / (i + base_q))

    # Pick curve function
    curve_fn = get_sine_curve if curve_mode == "sine" else get_cosine_curve

    # Projected coordinates u(X), v(X) for a given i
    def _u_of_X(X: float, i: int) -> float:
        y_old = float(curve_fn(X, i, N, p=p, lmbd=_lmbd_eff_for_i(i), linear=linear))
        rho = 1.0 - y_old
        return rho * np.cos(X)

    def _v_of_X(X: float, i: int) -> float:
        y_old = float(curve_fn(X, i, N, p=p, lmbd=_lmbd_eff_for_i(i), linear=linear))
        rho = 1.0 - y_old
        return rho * np.sin(X)    

    if curve_mode == "sine":
        # x_max from i=0  (maximize u)
        i_xmax = 0
        a, b = 0.0, np.pi
        x_at = fminbound(lambda X: -_u_of_X(X, i_xmax), a, b)
        x_max = _u_of_X(x_at, i_xmax)

        # x_min from i=0  (minimize u)
        i_xmin = 0
        a, b = 0.0, np.pi
        x_at = fminbound(lambda X: _u_of_X(X, i_xmin), a, b)
        x_min = _u_of_X(x_at, i_xmin)

        # y_max from i=0  (maximize v)
        i_ymax = 0
        a, b = 0.0, np.pi
        x_at = fminbound(lambda X: -_v_of_X(X, i_ymax), a, b)
        y_max = _v_of_X(x_at, i_ymax)

        # y_min from i=1  (minimize v)
        i_ymin = 1
        a, b = np.pi, np.pi*3/2
        x_at = fminbound(lambda X: _v_of_X(X, i_ymin), a, b)
        y_min = _v_of_X(x_at, i_ymin)

    else:  # curve_mode == "cosine"
        # x_max from i=0  (maximize u)
        i_xmax = 0
        a, b = 0.0, np.pi
        x_at = fminbound(lambda X: -_u_of_X(X, i_xmax), a, b)
        x_max = _u_of_X(x_at, i_xmax)

        # y_max from i=0  (maximize v)
        i_ymax = 0
        a, b = 0.0, np.pi
        x_at = fminbound(lambda X: -_v_of_X(X, i_ymax), a, b)
        y_max = _v_of_X(x_at, i_ymax)

        # x_min from i=1  (minimize u)
        x_min = 0
        for i_xmin in [0,1]:
            a, b = np.pi/2, np.pi*3/2
            x_at = fminbound(lambda X: _u_of_X(X, i_xmin), a, b)
            x_min = min(_u_of_X(x_at, i_xmin), x_min)

        # y_min from i=2  (minimize v)
        y_min = 0
        for i_ymin in [1,2]:
            a, b = np.pi*5/4, np.pi*7/4
            x_at = fminbound(lambda X: _v_of_X(X, i_ymin), a, b)
            y_min = min(_v_of_X(x_at, i_ymin), y_min)

    return float(x_min), float(x_max), float(y_min), float(y_max)
