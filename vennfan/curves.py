import numpy as np
from scipy.optimize import fminbound
from typing import Optional

# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------

def get_sine_curve(
    X,
    i: float,
    N: int,
    p: float = 0.33,
    decay: str = "linear",  # "linear" or "exponential"
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: float = 0.8,
    corrected: bool = False,
):
    """
    Compute a nonlinear, sine-based boundary curve for a given class index.

    The returned curve is based on a π-shifted sine:
        base(X) = sin(2**i * X - π)
    and then transformed as:
        curve(X) = amp(i) * sign(base) * |base|**p

    where the amplitude ``amp(i)`` depends on ``decay``:

    - ``decay="linear"``:
        * If ``N <= 2``: a simple linear ramp is used.
        * If ``N > 2``: amplitude is interpolated linearly from
          ``1 - epsilon`` at ``i=0`` to ``delta`` at ``i=N-2``.
          If ``epsilon``/``delta`` are not provided, they default to ``1/N``.
    - ``decay="exponential"``:
        amp(i) = b ** (i + epsilon)

    Special cases
    -------------
    - If ``i == N - 1``, returns zeros with the same shape as ``X``.
    - If ``corrected`` is True and ``i < 1``, positive-sign regions are clipped to 1:
        returns ``curve`` where ``sign(base) < 0``, else 1.

    Parameters
    ----------
    X : array_like
        Input angles (radians). Will be converted to a float NumPy array.
    i : float
        Class index (typically integer-valued). Controls frequency via ``2**i``.
    N : int
        Number of classes/levels. Used to determine amplitude schedules and the
        terminal case ``i == N - 1``.
    p : float, default=0.33
        Exponent applied to ``|sin|``. Must be positive for typical use.
    decay : {"linear", "exponential"}, default="linear"
        Amplitude schedule type.
    epsilon : float, optional
        For ``decay="linear"`` (when ``N > 2``): start offset controlling ``amp(0)=1-epsilon``.
        For ``decay="exponential"``: exponent offset in ``b**(i+epsilon)``.
    delta : float, optional
        For ``decay="linear"`` (when ``N > 2``): end value controlling the amplitude at ``i=N-2``.
        Ignored for ``decay="exponential"``.
    b : float, default=0.8
        Base for exponential decay. Only used when ``decay="exponential"``.
    corrected : bool, default=False
        If True, apply clipping behavior for small ``i`` as described above.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``X`` containing the boundary curve values.

    Raises
    ------
    ValueError
        If ``decay`` is not one of {"linear", "exponential"}.
    """
    X = np.asarray(X, float)

    if i == N - 1:
        return np.zeros_like(X)

    base = np.sin(2**i * X - np.pi)
    if decay == "linear":
        if N <= 2:
            amp = (N-1-i) / N
        else:
            if epsilon is None:
                epsilon = 1 / N
            if delta is None:
                delta = 1 / N
            amp = 1 - epsilon + i*(delta+epsilon-1) / (N-2)
    elif decay == "exponential":
        if epsilon is None:
            epsilon = 0.2
        amp = b ** (i+epsilon)
    else:
        raise ValueError("decay must be 'linear' or 'exponential'")

    sgn = np.sign(base)
    curve = amp * sgn * np.abs(base) ** p
    if i < 1 and corrected:
        return np.where(sgn < 0, curve, 1)
    return curve

def get_cosine_curve(
    X,
    i: float,
    N: int,
    p: float = 0.33,
    decay: str = "linear",  # "linear" or "exponential"
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: float = 0.8,
    corrected: bool = False,
):
    """
    Compute a nonlinear, cosine-based boundary curve for a given class index.

    The returned curve is based on a cosine with frequency scaling:
        base(X) = cos((2**i)/2 * (X + 2π))
    and then transformed as:
        curve(X) = amp(i) * sign(base) * |base|**p

    where the amplitude ``amp(i)`` depends on ``decay`` (same conventions as
    :func:`get_sine_curve`):

    - ``decay="linear"``:
        * If ``N <= 2``: a simple linear ramp is used.
        * If ``N > 2``: amplitude is interpolated linearly from
          ``1 - epsilon`` at ``i=0`` to ``delta`` at ``i=N-2``.
          If ``epsilon``/``delta`` are not provided, they default to ``1/N``.
    - ``decay="exponential"``:
        amp(i) = b ** (i + epsilon)

    Special cases
    -------------
    - If ``i == N - 1``, returns zeros with the same shape as ``X``.
    - If ``corrected`` is True and ``i < 2``, positive-sign regions are clipped to 1:
        returns ``curve`` where ``sign(base) < 0``, else 1.

    Parameters
    ----------
    X : array_like
        Input angles (radians). Will be converted to a float NumPy array.
    i : float
        Class index (typically integer-valued). Controls frequency via ``2**i``.
    N : int
        Number of classes/levels. Used to determine amplitude schedules and the
        terminal case ``i == N - 1``.
    p : float, default=0.33
        Exponent applied to ``|cos|``. Must be positive for typical use.
    decay : {"linear", "exponential"}, default="linear"
        Amplitude schedule type.
    epsilon : float, optional
        For ``decay="linear"`` (when ``N > 2``): start offset controlling ``amp(0)=1-epsilon``.
        For ``decay="exponential"``: exponent offset in ``b**(i+epsilon)``.
    delta : float, optional
        For ``decay="linear"`` (when ``N > 2``): end value controlling the amplitude at ``i=N-2``.
        Ignored for ``decay="exponential"``.
    b : float, default=0.8
        Base for exponential decay. Only used when ``decay="exponential"``.
    corrected : bool, default=False
        If True, apply clipping behavior for small ``i`` as described above.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``X`` containing the boundary curve values.

    Raises
    ------
    ValueError
        If ``decay`` is not one of {"linear", "exponential"}.
    """
    X = np.asarray(X, float)

    if i == N - 1:
        return np.zeros_like(X)

    base = np.cos(2**i / 2 * (X + 2*np.pi))
    if decay == "linear":
        if N <= 2:
            amp = (N-1-i) / N
        else:
            if epsilon is None:
                epsilon = 1 / N
            if delta is None:
                delta = 1 / N
            amp = 1 - epsilon + i*(delta+epsilon-1) / (N-2)
    elif decay == "exponential":
        if epsilon is None:
            epsilon = 0.2
        amp = b ** (i+epsilon)
    else:
        raise ValueError("decay must be 'linear' or 'exponential'")

    sgn = np.sign(base)
    curve = amp * sgn * np.abs(base) ** p
    if i < 2 and corrected:
        return np.where(sgn < 0, curve, 1)
    return curve

def vennfan_find_extrema(
    curve_mode: str,
    p: float,
    decay: str = "linear",  # "linear" or "exponential"
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    b: float = 0.8,
    N: int = 6,
) -> tuple[float, float, float, float]:
    """
    Find extrema of the disc-projected class-boundary curves used by the vennfan mapping.

    For a selected boundary curve y_old(X) (sine- or cosine-based), this function maps
    points to the disc using
        rho(X) = 1 - y_old(X)
        u(X)   = rho(X) * cos(X)
        v(X)   = rho(X) * sin(X)
    and numerically estimates the extrema (x_min, x_max, y_min, y_max) of u and v by
    bounded 1D optimization over fixed sub-intervals of X.

    The boundary curve y_old(X) is computed by :func:`get_sine_curve` or
    :func:`get_cosine_curve` (selected by ``curve_mode``). The parameters ``p``,
    ``decay``, ``epsilon``, ``delta``, ``b``, and ``N`` are passed through to the
    underlying curve function unchanged.

    Search strategy (heuristics)
    ----------------------------
    To reduce the number of optimizations, extrema are searched using only a small set
    of curve indices ``i`` and restricted X-ranges:

    - ``curve_mode == "sine"``
        * x_max: maximize u(X) using i=0 over X ∈ [0, π]
        * x_min: minimize u(X) using i=0 over X ∈ [0, π]
        * y_max: maximize v(X) using i=0 over X ∈ [0, π]
        * y_min: minimize v(X) using i=1 over X ∈ [π, 3π/2]

    - ``curve_mode == "cosine"``
        * x_max: maximize u(X) using i=0 over X ∈ [0, π]
        * y_max: maximize v(X) using i=0 over X ∈ [0, π]
        * x_min: minimize u(X) using i ∈ {0, 1} over X ∈ [π/2, 3π/2] (take the minimum)
        * y_min: minimize v(X) using i ∈ {1, 2} over X ∈ [5π/4, 7π/4] (take the minimum)

    Parameters
    ----------
    curve_mode : {"sine", "cosine"}
        Selects the underlying boundary curve family.
    p : float
        Exponent applied to |sin| or |cos| inside the boundary curve. Must be > 0.
    decay : {"linear", "exponential"}, default="linear"
        Amplitude schedule passed through to the underlying curve function.
    epsilon : float, optional
        Passed through to the underlying curve function (meaning depends on ``decay``).
    delta : float, optional
        Passed through to the underlying curve function for linear decay (N > 2).
    b : float, default=0.8
        Exponential decay base passed through to the underlying curve function.
    N : int, default=6
        Number of classes/levels used by the underlying curve function.

    Returns
    -------
    tuple[float, float, float, float]
        (x_min, x_max, y_min, y_max), where x_* are extrema of u(X) and y_* are extrema
        of v(X) under the heuristic search described above.

    Raises
    ------
    ValueError
        If ``p <= 0`` or if ``curve_mode`` is not one of {"sine", "cosine"}.

    Notes
    -----
    - Extrema are approximations produced by ``scipy.optimize.fminbound`` on the
      specified intervals; they are not guaranteed to be global extrema over X ∈ [0, 2π].
    - The chosen i-values and intervals encode assumptions about where the relevant
      extrema occur for the vennfan construction.
    """
    if p <= 0:
        raise ValueError("p must be > 0.")
    if curve_mode not in ("sine", "cosine"):
        raise ValueError("curve_mode must be 'sine' or 'cosine'.")

    # Pick curve function
    curve_fn = get_sine_curve if curve_mode == "sine" else get_cosine_curve

    # Projected coordinates u(X), v(X) for a given i
    def _u_of_X(X: float, i: int) -> float:
        y_old = float(curve_fn(X, i, N, p=p, decay=decay, epsilon=epsilon, delta=delta, b=b))
        rho = 1.0 - y_old
        return rho * np.cos(X)

    def _v_of_X(X: float, i: int) -> float:
        y_old = float(curve_fn(X, i, N, p=p, decay=decay, epsilon=epsilon, delta=delta, b=b))
        rho = 1.0 - y_old
        return rho * np.sin(X)

    if curve_mode == "sine":
        # x_max from i=0  (maximize u)
        i_xmax = 0
        x_at = fminbound(lambda X: -_u_of_X(X, i_xmax), 0.0, np.pi)
        x_max = _u_of_X(x_at, i_xmax)

        # x_min from i=0  (minimize u)
        i_xmin = 0
        x_at = fminbound(lambda X: _u_of_X(X, i_xmin), 0.0, np.pi)
        x_min = _u_of_X(x_at, i_xmin)

        # y_max from i=0  (maximize v)
        i_ymax = 0
        x_at = fminbound(lambda X: -_v_of_X(X, i_ymax), 0.0, np.pi)
        y_max = _v_of_X(x_at, i_ymax)

        # y_min from i=1  (minimize v)
        i_ymin = 1
        x_at = fminbound(lambda X: _v_of_X(X, i_ymin), np.pi, np.pi*3/2)
        y_min = _v_of_X(x_at, i_ymin)

    else:  # curve_mode == "cosine"
        # x_max from i=0  (maximize u)
        i_xmax = 0
        x_at = fminbound(lambda X: -_u_of_X(X, i_xmax), 0.0, np.pi)
        x_max = _u_of_X(x_at, i_xmax)

        # y_max from i=0  (maximize v)
        i_ymax = 0
        x_at = fminbound(lambda X: -_v_of_X(X, i_ymax), 0.0, np.pi)
        y_max = _v_of_X(x_at, i_ymax)

        # x_min from i=1  (minimize u)
        x_min = 0
        for i_xmin in [0,1]:
            x_at = fminbound(lambda X: _u_of_X(X, i_xmin), np.pi/2, np.pi*3/2)
            x_min = min(_u_of_X(x_at, i_xmin), x_min)

        # y_min from i=2  (minimize v)
        y_min = 0
        for i_ymin in [1,2]:
            x_at = fminbound(lambda X: _v_of_X(X, i_ymin), np.pi*5/4, np.pi*7/4)
            y_min = min(_v_of_X(x_at, i_ymin), y_min)

    return float(x_min), float(x_max), float(y_min), float(y_max)
