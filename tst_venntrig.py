import os
from vennfan import venntrig, make_demo_values
import numpy as np

if __name__ == "__main__":
    os.makedirs("img/venntrig/nolabels/", exist_ok=True)
    greek_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
        "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]

    for curve_mode in ["cosine", "sine"]:
        for decay in ["linear", "exponential"]:
            for N in range(1, 10):
                print(
                    f"Generating venntrig diagram for "
                    f"curve_mode={curve_mode} decay={decay} N={N} ..."
                )
                values = make_demo_values(N)
                class_names = greek_names[:N]

                outfile = f"img/venntrig/{curve_mode}_{decay}_N{N}.png"
                venntrig(
                    values,
                    class_names,
                    outfile=outfile,
                    height_scale=2.0,
                    p=0.2,
                    decay=decay,
                    epsilon=None,
                    delta=None,
                    b=0.8,
                    include_constant_last=True,
                    curve_mode=curve_mode,
                    color_mixing="average",
                    region_label_fontsize=20,
                    adaptive_fontsize=False
                )

    for curve_mode in ["cosine", "sine"]:
        for decay in ["linear", "exponential"]:
            for N in range(1, 10):
                print(
                    f"Generating venntrig diagram for "
                    f"curve_mode={curve_mode} decay={decay} N={N} ..."
                )

                values = np.empty((2,) * N, dtype=object)

                outfile = f"img/venntrig/nolabels/{curve_mode}_{decay}_N{N}.png"
                venntrig(
                    values=values,
                    outfile=outfile,
                    height_scale=2.0,
                    p=0.2,
                    decay=decay,
                    epsilon=None,
                    delta=None,
                    b=0.8,
                    include_constant_last=True,
                    curve_mode=curve_mode,
                    color_mixing="average",
                    region_label_fontsize=20,
                    adaptive_fontsize=False
                )
