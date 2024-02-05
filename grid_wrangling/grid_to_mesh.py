from pathlib import Path
from typing import Tuple, Union

import numpy as np

"""
Convert standard Emu grid files into Ansys meshes. The resulting elements are of type
SOLID185 and are cubes centered on the points of the grid. Coincident nodes in the
resulting mesh are merged.

Each mesh is saved as two files: nodes and elements. These files inherit the name and
location of the grid file and are given the extensions .node and .elem, respectively.

Usage:
The first argument is the edge length of the resulting cubic elements. For a regular
grid, this is the same as the grid spacing.

All following arguments are paths to grid files to be converted. An arbitrary number can
be specified, and they all use the same edge length.

Example:
> python grid_to_mesh.py 1.0 example1.grid .\grid_files\example2.grid

Note:
"Standard Emu grid file" means a whitespace delimited file where the first row is the
number of nodes and all following rows contain exactly four columns: x, y, z, material.
"""

# fmt: off
# According to the node order of the SOLID185 element type
OFFSETS = (
    ( 1, -1, -1),
    ( 1,  1, -1),
    (-1,  1, -1),
    (-1, -1, -1),
    ( 1, -1,  1),
    ( 1,  1,  1),
    (-1,  1,  1),
    (-1, -1,  1),
)
# fmt: on


def from_file(grid_path: Union[str, Path], edge_length: float) -> None:
    coords, mats = load_grid(grid_path)

    save_folder = Path(grid_path).resolve().parent

    convert(
        coords,
        edge_length,
        save_folder,
        mats,
        csys=None,
        filename_override=Path(grid_path).stem,
    )


def load_grid(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a standard Emu grid file into numpy arrays.

    Args:
        file_path (Union[str, Path]): path to grid file

    Returns:
        nodes (np.ndarray): (n x 3) array of node coordinates
        mats (np.ndarray): (n) array of node materials
    """
    assert Path(file_path).exists(), f"Grid file not found at {file_path}"

    with Path(file_path).open() as grid_file:
        # *var1, var2 means put the last column in var2 and all the others in var1
        *nodes, mats = np.genfromtxt(
            grid_file, delimiter=" ", skip_header=1, unpack=True
        )

        return np.transpose(nodes), mats


def convert(
    coords, edge_length, save_folder, mats, csys=None, filename_override: str = None
):
    element_node_coords = offset_cube_centroids_to_corners(coords, edge_length)

    nodes_unique, node_nums, element_node_nums = assign_unique_nodes(
        element_node_coords
    )

    name = filename_override or "mesh"

    write_nodes(nodes_unique, node_nums, Path(save_folder) / f"{name}.node")
    write_elements(element_node_nums, mats, Path(save_folder) / f"{name}.elem", csys)


def offset_cube_centroids_to_corners(
    centroids: np.ndarray, side_length: float
) -> np.ndarray:
    """Convert cube centroids into the coordinates of their corners.

    Args:
        centroids (np.ndarray): (n x 3) array containing coordinates of cube centroids
        side_length (float): side length of the cube(s)

    Returns:
        np.ndarray: (n x 8 x 3) array containing coordinates of each cube corner
    """
    return np.stack(
        [centroids + offset * side_length / 2 for offset in np.array(OFFSETS)],
        axis=1,
    )


def assign_unique_nodes(element_node_coords: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Eliminate the duplicate nodes shared by elements.

    Args:
        element_node_coords (np.ndarray): (e x 8 x 3) array containing x,y,z coordinates

    Returns:
        nodes_unique (np.ndarray): (n x 3) array containing unique x,y,z coordinates
        node_nums (np.ndarray): (n) array containing new contiguous node numbers
        element_node_nums (np.ndarray): (e x 8) array of element nodes with new numbers
    """
    # Flatten the element nodes array into a list of all nodes
    flattened_node_coords = element_node_coords.reshape((-1, 3))
    # Get the unique coordinates and the indices at which those coordinates appear
    nodes_unique, unique_inverse = np.unique(
        flattened_node_coords, axis=0, return_inverse=True
    )
    node_nums = np.arange(1, len(nodes_unique) + 1)  # New contiguous node numbers

    # Invert the `unique` operation, but with the node numbers instead of coords
    # Un-flatten the list back into the element nodes list
    element_node_nums = node_nums[unique_inverse].reshape((-1, 8))

    return (nodes_unique, node_nums, element_node_nums)


def write_nodes(
    node_coords: np.ndarray, node_nums: np.ndarray, node_path: Union[str, Path]
) -> None:
    """Write a set of nodes to file in Ansys APDL format.
    From APDL documentation:
    The format used is (I9, 6G21.13E3) to write out NODE,X,Y,Z,THXY,THYZ,THZX. If the
    last number is zero (THZX = 0), or the last set of numbers are zero, they are not
    written but are left blank.

    Args:
        node_coords (np.ndarray): (n x 3) array containing x,y,z coordinates
        node_nums (np.ndarray): (n) array containing node numbers
        node_path (Union[str, Path]): path to file to save, including name and extension
    """
    assert len(node_coords) == len(node_nums)
    with Path(node_path).open(mode="w") as node_file:
        np.savetxt(
            node_file,
            np.concatenate([node_nums[:, None], node_coords], axis=-1),
            fmt=["%9i", "%21.13e", "%21.13e", "%21.13e"],
            delimiter="",
        )


def write_elements(
    elem_node_nums: np.ndarray,
    elem_mats: np.ndarray,
    elem_path: Union[str, Path],
    elem_csys: np.ndarray = None,
) -> None:
    """Write a set of elements to file in Ansys APDL format.
    From APDL documentation:
    The data description of each record is:
        I, J, K, L, M, N, O, P, MAT, TYPE, REAL, SECNUM, ESYS, IEL
    The format is (14I6) if Format = SHORT, and (14I8 or 14I10) if Format = LONG.

    Args:
        elem_node_nums (np.ndarray): (e x 8) array containing node numbers for each element
        elem_mats (np.ndarray): (e x 1) array containing material number for each element
        elem_path (Union[str, Path]): path to file to save, including name and extension
    """
    assert len(elem_node_nums) == len(elem_mats)
    num_elems = len(elem_node_nums)
    type_dummy = np.ones((num_elems, 1))
    real_dummy = np.ones((num_elems, 1))
    secnum_dummy = np.ones((num_elems, 1))
    iel = np.arange(1, num_elems + 1)
    if elem_csys is None:
        elem_csys = np.zeros((num_elems))

    elem_data = np.concatenate(
        [
            elem_node_nums,
            elem_mats[:, None],
            type_dummy,
            real_dummy,
            secnum_dummy,
            elem_csys[:, None],
            iel[:, None],
        ],
        axis=-1,
    )

    with Path(elem_path).open(mode="w") as elem_file:
        np.savetxt(
            elem_file,
            elem_data,
            fmt="%8i",
            delimiter="",
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Not enough arguments")
        exit()

    edge_length = float(sys.argv[1])
    grid_files = (Path(arg) for arg in sys.argv[2:])

    for grid_file in grid_files:
        print(f"Converting {grid_file} ...")
        from_file(grid_file, edge_length)
