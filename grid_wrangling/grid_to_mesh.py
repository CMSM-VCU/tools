from pathlib import Path
from typing import Tuple, Union

import numpy as np

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


def from_file(
    grid_path: Union[str, Path],
    edge_length: float,
    save_folder: Union[str, Path] = None,
) -> None:
    coords, mats = load_grid(grid_path)

    if save_folder is None:
        save_folder = Path(grid_path).resolve().parent
    else:
        save_folder = Path(save_folder)

    convert(coords, edge_length, save_folder, mats, None)


def convert(
    coords, edge_length, save_folder, mats, csys=None, filename_override: str = None
):
    offset_coords = np.stack(
        offset_cube_centroids_to_corners(coords, edge_length), axis=1
    )
    nodes_unique, node_nums, element_node_nums = assign_unique_nodes(
        offset_coords.reshape((-1, 3))
    )

    name = filename_override or "mesh"

    write_nodes(nodes_unique, node_nums, Path(save_folder) / f"{name}.node")
    write_elements(element_node_nums, mats, Path(save_folder) / f"{name}.elem", csys)


def load_grid(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    assert Path(file_path).exists(), f"Grid file not found at {file_path}"
    with Path(file_path).open() as grid_file:
        *nodes, mats = np.genfromtxt(
            grid_file, delimiter=",", skip_header=1, unpack=True
        )

        return np.transpose(nodes), mats


def offset_cube_centroids_to_corners(
    centroids: np.ndarray, side_length: float
) -> Tuple[np.ndarray, ...]:
    return tuple(centroids + offset * side_length / 2 for offset in np.array(OFFSETS))


def assign_unique_nodes(nodes: np.ndarray) -> Tuple[np.ndarray, ...]:
    nodes_unique, unique_inverse = np.unique(nodes, axis=0, return_inverse=True)
    node_nums = np.arange(1, len(nodes_unique) + 1)

    element_node_nums = node_nums[unique_inverse].reshape((-1, 8))
    return (nodes_unique, node_nums, element_node_nums)


def write_nodes(
    node_coords: np.ndarray, node_nums: np.ndarray, node_path: Union[str, Path]
) -> None:
    """ Write a set of nodes to file in Ansys APDL format.
    From APDL documentation:
    The format used is (I9, 6G21.13E3) to write out NODE,X,Y,Z,THXY,THYZ,THZX. If the
    last number is zero (THZX = 0), or the last set of numbers are zero, they are not
    written but are left blank.

    Args:
        node_coords (np.ndarray): (n x 3) array containing x,y,z coordinates
        node_nums (np.ndarray): (n x 1) array containing node numbers
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
            elem_file, elem_data, fmt="%8i", delimiter="",
        )


if __name__ == "__main__":
    folder = Path("./outputs/debug/3x4x5_p0.0_c0.25,2_r1.0_1/")
    grid_path = folder / "grid.csv"
    edge_length = 1.0

    from_file(grid_path, edge_length)
    pass
