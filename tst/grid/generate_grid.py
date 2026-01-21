import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Make matplotlib safe for multiprocessing (no GUI backend)
os.environ.setdefault("MPLBACKEND", "Agg")

from vennfan import vennfan, make_demo_values


def _render_one(N: int, curve_mode: str, p: float, b: float, greek_names: list[str], outdir: str, highlight_factor=None):
    import numpy as _np
    import matplotlib.pyplot as _plt
    if curve_mode == "sine":
        if N > 2 and p == 1 and b > 2/3:
            return None  # skip invalid combinations
        if N > 2 and p == 1/2 and b > 3/4:
            return None  # skip invalid combinations
        if N > 2 and p == 1/4 and b > 7/8:
            return None  # skip invalid combinations
    elif curve_mode == "cosine":
        if N > 5 and p == 1 and b > 2/3:
            return None  # skip invalid combinations
        if N == 5 and p == 1 and b > 3/4:
            return None  # skip invalid combinations
        if N == 4 and p == 1 and b > 5/6:
            return None  # skip invalid combinations
        if N > 5 and p == 1/2 and b > 4/5:
            return None  # skip invalid combinations
        if N == 5 and p == 1/2 and b > 5/6:
            return None  # skip invalid combinations
        if N > 7 and p == 1/4 and b > 7/8:
            return None  # skip invalid combinations

    _ = make_demo_values(N)  # kept to preserve original intent; not used below
    values = _np.empty((2,) * N, dtype=object)

    class_names = greek_names[:N]
    class_names = [""] * N  # hide class labels in this demo if desired
    os.makedirs(f"{outdir}/N{N}", exist_ok=True)
    outfile = (
        f"{outdir}/N{N}/vennfan_{curve_mode}_N{N}_p{p:.3g}_b{b:.3g}_exponential.svg"
    )

    vennfan(
        values,
        class_names,
        outfile=outfile,
        decay="exponential",
        p=p,
        b=b,
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

    p_values = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
    b_values = [1/2, 2/3, 3/4, 4/5, 5/6, 7/8, 9/10]
    curve_modes = ["sine", "cosine"]

    for N in range(2, 9):
        for curve_mode in curve_modes:
            pairs = [(p, b) for p in p_values for b in b_values]
            desc = f"curve_mode={curve_mode} N={N}"
            if N < 8:
                workers = 32
            else:
                workers = 8
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(_render_one, N, curve_mode, p, b, greek_names, outdir, highlight_factor)
                    for (p, b) in pairs
                ]

                with tqdm(total=len(futures), desc=desc, unit="pair") as pbar:
                    for fut in as_completed(futures):
                        fut.result()  # raise exceptions promptly
                        pbar.update(1)
