import sys
sys.path.append("./vennfan")

import os
from utils import make_demo_values
from vennfan import vennfan
import numpy as np

if __name__ == "__main__":
    os.makedirs("img/vennfan/", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    for curve_mode in ["sine", "cosine"]:
        for lmbd in [1 / 6]:
            for N in range(4, 5):  # tweak as desired
                print(
                    f"Generating vennfan diagram for "
                    f"curve_mode={curve_mode} linear_decay={lmbd is None} N={N}..."
                )
                shape = (2,) * N
                values = np.empty(shape, dtype=object)

                # Demo values: label each region by subset of A,B,C,...
                values = make_demo_values(N)
                class_names = greek_names[:N]
                class_names = [""] * N  # hide class labels in this demo if desired

                outfile = f"img/vennfan/vennfan_{curve_mode}{'_linear' if lmbd is None else ''}_N{N}.pdf"
                vennfan(
                    values,
                    class_names,
                    outfile=outfile,
                    p=1 / 7,
                    epsilon=1 / 8,
                    curve_mode=curve_mode,
                    draw_tight_factor=1.02,
                    color_mixing="average",
                    region_label_placement="visual_text_center",
                    radial_bias=0.6,
                    region_fontsize=10,
                    radial_region_fontsize=10,
                    visual_center_rotate_toward_radial=True,
                    visual_text_center_area_fraction=0.15,
                    highlight_colors=0.75,
                    text_color="black"
                )
 