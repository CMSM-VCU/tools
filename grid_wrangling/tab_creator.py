import numpy as np
from grid import Grid


def create_tabs(grid:Grid, axis:int, positive_mat:int=1, negative_mat:int=2):
    if 1 > axis > 3: 
        raise Exception("Axis must be 1, 2, or 3")
    elif 1 > positive_mat > 999 or 1 > negative_mat > 999:
        raise Exception("Material number must be greater than 0 (not inclusive) and less than 1000 (not inclusive)")

    extents = grid.extents()
    lengths = grid.lengths()
    grid_spacing = grid.grid_spacing()[1]
    num_points = (lengths / grid_spacing).astype(np.int32) + 1

    num_points[axis-1] = 3
    x,y,z = np.mgrid[0:num_points[0], 0:num_points[1], 0:num_points[2]]
    coords = np.vstack((x.flatten(),y.flatten(),z.flatten())).T

    base_tab = Grid(coords=coords, mats=np.ones_like(coords[:,0]).astype(np.int16))
    base_tab.center_and_normalize()
    base_tab.scale(np.amax(extents))

    translation = np.zeros(3)
    translation[axis-1] = extents[:,1][axis-1] + 2 * grid_spacing

    positive_tab = base_tab.copy()
    positive_tab.translate(translation)
    positive_tab.set_mats(positive_tab.mats * positive_mat)

    negative_tab = base_tab.copy()
    negative_tab.translate(-translation)
    negative_tab.set_mats(negative_tab.mats * negative_mat)

    return positive_tab.append(negative_tab)





if __name__ == "__main__":
    grid = Grid(fname="example.grid")
    tabs = create_tabs(grid, 2)
    grid.append(tabs).visualize()



