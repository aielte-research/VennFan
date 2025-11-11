import numpy as np

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
):
    """
    Nonlinear sine-like curve for class boundary (π-shifted).

    p : exponent on |sin|
    lmbd : amplitude decay factor in log-frequency
    """
    X = np.asarray(X, float)
    base = np.sin(2**i * X - np.pi)
    if linear:
        amp = (N-i-1) / N
    else:
        amp = lmbd ** (i+1)
    
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
):
    """
    Nonlinear cosine-like curve for class boundary (π-shifted).

    p : exponent on |sin|
    lmbd : amplitude decay factor in log-frequency
    """
    X = np.asarray(X, float)

    base = np.cos(- 2**i/2 * (X+2*np.pi))
    if linear:
        amp = (N-i-1) / N
    else:
        amp = lmbd ** (i+1)
        
    sgn = np.sign(base)
    curve = amp * sgn * np.abs(base) ** p
    if i<2 and corrected:
        return np.where(sgn<0, curve, 1)
    return curve
