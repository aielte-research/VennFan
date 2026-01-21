import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Make matplotlib safe for multiprocessing (no GUI backend)
os.environ.setdefault("MPLBACKEND", "Agg")

from vennfan import vennfan, make_demo_values


def _render_one(N: int, curve_mode: str, p: float, eps: float, delta: float, greek_names: list[str], outdir: str, highlight_factor=None):
    import numpy as _np
    import matplotlib.pyplot as _plt

    if N < 3 and not (p == 1/3 and eps == 0 and delta == 1/8):
        return None  # skip invalid combinations
    
    if curve_mode == "sine":
        if N == 8 and p == 1/3:
            return None  # skip invalid combinations
        if N == 8 and p == 1/4 and delta > 1/4:
            return None  # skip invalid combinations
        if N == 7 and p == 1/3 and delta > 1/6:
            return None  # skip invalid combinations
        if N == 7 and p == 1/4 and delta > 1/3:
            return None  # skip invalid combinations
        if N == 6 and p == 1/3 and delta > 1/3:
            return None  # skip invalid combinations
    elif curve_mode == "cosine":
        if N == 8 and p == 1/3 and delta > 1/4:
            return None  # skip invalid combinations
        if N == 8 and p == 1/4 and delta > 1/3:
            return None  # skip invalid combinations
        if N == 7 and p == 1/3 and delta > 1/3:
            return None  # skip invalid combinations

    _ = make_demo_values(N)  # kept to preserve original intent; not used below
    values = _np.empty((2,) * N, dtype=object)

    class_names = greek_names[:N]
    class_names = [""] * N  # hide class labels in this demo if desired
    os.makedirs(f"{outdir}/N{N}", exist_ok=True)
    outfile = (
        f"{outdir}/N{N}/vennfan_{curve_mode}_N{N}_p{p:.3g}_epsilon{eps:.3g}_delta{delta:.3g}_linear.svg",
    )

    vennfan(
        values,
        class_names,
        outfile=outfile,
        decay="linear",
        p=p,
        epsilon=eps,
        delta=delta,
        curve_mode=curve_mode,
        draw_tight_factor=1.02,
        color_mixing="average",
        region_label_placement="visual_text_center",
        radial_bias=0.6,
        visual_center_rotate_toward_radial=True,
        visual_text_center_area_fraction=0.15,
        text_color="black",
        highlight_factor=highlight_factor
    )

    _plt.close("all")
    return outfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Enable highlight colors (passes highlight_factor=0.75).",
    )
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if args.highlight:
        outdir = "img/vennfan_grid_highlight"
        highlight_factor = 0.75
    else:
        outdir = "img/vennfan_grid"
        highlight_factor = None
    os.makedirs(outdir, exist_ok=True)

    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    p_values = [1/3, 1/4, 1/8, 1/16, 1/32, 1/64]
    eps_values = [0, 1/16, 1/8, 1/4, 1/2]
    delta_values = [1/8, 1/6, 1/4, 1/3, 1/2]
    curve_modes = ["sine", "cosine"]

    workers = int(os.environ.get("VENNFAN_WORKERS", "32"))

    for N in range(2, 9):
        for curve_mode in curve_modes:
            pairs = [(p, eps, delta) for p in p_values for eps in eps_values for delta in delta_values]
            desc = f"curve_mode={curve_mode} N={N}"
            if N < 8:
                workers = 32
            elif N == 8:
                workers = 16
            else:
                workers = 8
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(_render_one, N, curve_mode, p, eps, delta, greek_names, outdir, highlight_factor)
                    for (p, eps, delta) in pairs
                ]

                with tqdm(total=len(futures), desc=desc, unit="pair") as pbar:
                    for fut in as_completed(futures):
                        fut.result()  # raise exceptions promptly
                        pbar.update(1)
