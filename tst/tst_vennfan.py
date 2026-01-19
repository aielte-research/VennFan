import os
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Make matplotlib safe for multiprocessing (no GUI backend)
os.environ.setdefault("MPLBACKEND", "Agg")

from vennfan import vennfan, make_demo_values


def _render_one(curve_mode: str, decay: str, N: int, greek_names: list[str], outdir: str) -> str:
    # Keep imports inside to reduce parent-process state leakage
    import matplotlib.pyplot as plt

    values = make_demo_values(N)
    class_names = greek_names[:N]
    outfile = os.path.join(outdir, f"vennfan_{curve_mode}_{decay}_{N}.png")

    vennfan(
        values,
        class_names,
        outfile=outfile,
        decay=decay,
        curve_mode=curve_mode,
        draw_tight_factor=1.02,
        color_mixing="average",
        region_label_placement="visual_text_center",
        radial_bias=0.6,
        visual_center_rotate_toward_radial=True,
        visual_text_center_area_fraction=0.15,
        text_color="black",
    )

    plt.close("all")
    return outfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Default 1 disables multiprocessing.",
    )
    args = parser.parse_args()
    workers = max(1, args.workers)

    outdir = "img/vennfan"
    os.makedirs(outdir, exist_ok=True)

    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    tasks: list[tuple[str, str, int]] = [
        (curve_mode, decay, N)
        for curve_mode in ["sine", "cosine"]
        for decay in ["linear", "exponential"]
        for N in range(2, 10)
    ]

    if workers == 1:
        # No multiprocessing
        for curve_mode, decay, N in tasks:
            print(f"Generating vennfan diagram for curve_mode={curve_mode} decay={decay} N={N}...")
            _render_one(curve_mode, decay, N, greek_names, outdir)
        return

    # Multiprocessing
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        future_to_task = {
            ex.submit(_render_one, curve_mode, decay, N, greek_names, outdir): (curve_mode, decay, N)
            for (curve_mode, decay, N) in tasks
        }

        for fut in as_completed(future_to_task):
            curve_mode, decay, N = future_to_task[fut]
            # Surface exceptions with task context
            try:
                outfile = fut.result()
                print(f"Done: curve_mode={curve_mode} decay={decay} N={N} -> {outfile}")
            except Exception as e:
                raise RuntimeError(f"Failed: curve_mode={curve_mode} decay={decay} N={N}") from e


if __name__ == "__main__":
    main()
