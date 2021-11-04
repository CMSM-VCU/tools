import sys
from pathlib import Path

import pandas as pd


def main(gridfile: Path):
    grid = pd.read_csv(
        gridfile,
        skiprows=1,
        names=["x1", "x2", "x3", "material"],
        sep=None,
        skipinitialspace=True,
        engine="python",
    )
    grid["iter"] = 0
    grid["m_global"] = grid.index
    grid.set_index(["iter", "m_global"], inplace=True)
    grid["u1"] = 0
    grid["u2"] = 0
    grid["u3"] = 0

    grid.to_hdf(gridfile.with_suffix(".h5"), "data", "w", complevel=9, format="table")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        gridfile = Path(sys.argv[1])
    else:
        print("No path specified")
        exit(0)

    main(gridfile=gridfile)
