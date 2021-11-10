""" Extracts and compresses data from Emu plot files.
Plot files must be in CSV format. The first row of each plot file must be the column titles.

Plot files must be name [filename].[timestep].[processor]
"""
import sys
from pathlib import Path

import h5py
import pandas as pd


def discard_empty_files(filenames):
    not_empty_filenames = list(filenames)
    for i, filename in reversed(list(enumerate(filenames))):
        with open(filename, mode="r") as f:
            f.readline()  # Dump header line
            if not f.readline():
                not_empty_filenames.remove(i)

    return not_empty_filenames


def load_plot_files(filenames):
    for filename in filenames:
        yield pd.read_csv(
            filename, index_col=["iter", "m_global"], skipinitialspace=True
        )


def main(search_path, filename_base):
    target_file = search_path / "simulation.h5"

    plotfiles = list(search_path.glob(f"{filename_base}.*.*"))

    not_empty_plotfiles = discard_empty_files(plotfiles)

    data = pd.concat(load_plot_files(not_empty_plotfiles))
    data = data.dropna(how="all", axis="columns")  # Drop column made by trailing commas

    data.to_hdf(target_file, "data", "w", complevel=9, format="table")

    with h5py.File(target_file, "a") as h5file:
        dt = h5py.special_dtype(vlen=str)
        readme = h5file.create_dataset("README", shape=(1,), dtype=dt)
        readme[0] = (
            "The data group of this h5 file was generated using the Pandas hdf5 functionality. "
            + "It can be decoded into a Pandas DataFrame using the pandas.read_hdf() function."
        )

    print(
        f"Plot data saved in {target_file}"
        + f"\n\tMax node number: {data.index.max()[1]}"
        + f"\n\tLast time step: {data.index.max()[0]}"
    )


if __name__ == "__main__":
    search_path = Path(".")
    filename_base = "emu.plt"
    if len(sys.argv) >= 2:
        search_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        filename_base = str(sys.argv[2])

    main(search_path, filename_base)
