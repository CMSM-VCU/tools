import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mode

class Grid:
    def __init__(self, fname=None, coords=None, mats=None):
        """Create a grid, read in from file name or create with specified coordinates and materials

        Args:
            fname (str, optional): Filename of .grid file to read from. Defaults to None.
            coords (np.ndarray, optional): 3D array of coordinates where each column represents a dimension. Defaults to None.
            mats (np.ndarray, optional): 1D array of material numbers. Defaults to None.

        Raises:
            Exception: Must include either (filename) or (coordinates and materials)
        """
        self.coords = None
        self.mats = None

        if fname is not None:
            data = np.loadtxt(fname, skiprows=1)
            self.set_coords(data[:, :3])
            self.set_mats(data[:, 3])
            pass
        
        elif coords is not None and mats is not None:
            self.set_coords(coords)
            self.set_mats(mats)
        
        else:
            raise Exception("Must have (fname) or (coords and mats)")

    def set_coords(self, coords):
        """Sets coordinates to given coordinates with some input checking

        Args:
            coords (3D Numpy Array): Each row is a point with column 1 in x1 column 2 in x2 and column 3 in x3

        Raises:
            Exception: If given coordinates are not a numpy array
            Exception: If given coordinates are not 3 dimensional
        """

        if type(coords) is not np.ndarray:
            raise Exception("Coords needs to be a numpy array")
        elif coords.shape[1] != 3:
            raise Exception("Coordiantes must be 3 dimensional")
        else:
            self.coords = coords

    def set_mats(self, mats):
        """Sets this grid's materials to the given materials

        Args:
            mats (1D Numpy array): Each material will be converted to a 16 bit signed int

        Raises:
            Exception: If given material numbers are not a numpy array
        """

        if type(mats) is not np.ndarray:
            raise Exception("Mats needs to be a numpy array")
        else:
            self.mats = mats.astype(np.int16)
            if np.amin(mats <= 0):
                print("WARNING EMU DOES NOT SUPPORT NEGATIVE OR ZERO MATERIAL NUMBERS")
            if np.amax(mats > 999):
                print("WARNING EMU DOES NOT SUPPORT MATERIAL NUMBERS GREATER THAN 999")

    def write(self, fname, format="%-28.18e", mat_format="%-3d"):
        """Write the grid to a file

        Args:
            fname (str): Name of the file to save as
            format (str, optional): Format for writing the coordinates. Defaults to "%-28.18e".
            mat_format (str, optional): Format for writing the materials. Defaults to "%-3d".
        """

        f = open(fname, 'w')
        f.write(str(self.coords.shape[0]) + "\n")
        for i in range(self.coords.shape[0]):
            f.write(format % (self.coords[i, 0]) + format % (self.coords[i, 1]) + format % (self.coords[i, 2]) + mat_format % (self.mats[i]) + "\n")
        f.close()        

    def extents(self): 
        """Calculates the extrema of the grid

        Returns:
            3x2 array: First column is the lower bound, the second column is the upper bound
        """

        return np.vstack((np.amin(self.coords, axis=0), np.amax(self.coords, axis=0))).T

    def grid_spacing(self):
        """Uses the nearest neighbor of each point to calculate the grid spacing

        Returns:
            (float, float): (minimum of nearest neighbors, mode of nearest neighbors) 
        """

        tree = cKDTree(self.coords)
        distances, _ = tree.query(self.coords, k=2, workers=-1)
        distances = distances[:, 1]
        return np.amin(distances), mode(distances)[0][0]

    def translate(self, vector):
        """Translates the grid in 3D space

        Args:
            vector (array): Amount to translate
        """

        self.coords = self.coords + vector

    def scale(self, vector):
        """Scales the grid in 3D space

        Args:
            vector (array): Amount to scale the grid by
        """

        self.coords = self.coords * vector

    def center_grid_bb(self):
        """Translates the grid so its center as defined by it's extents or bounding box is 0,0,0
        """
        self.translate(-self.center_of_bounding_box())

    def center_grid_ap(self):
        """Translates the grid so its center as defined by the average point position is 0,0,0
        """
        self.translate(-self.center_from_average_position())

    def center_of_bounding_box(self):
        """Calculates the center of the bounding box (also the center of the extents)

        Returns:
            array: centerx1, centerx2, centerx3
        """

        extents = self.extents()
        return np.average(extents, axis=1)

    def center_from_average_position(self):
        """Calculates the average position of the grid

        Returns:
            array: centerx1, centerx2, centerx3
        """

        return np.average(self.coords, axis=0)

    def collapse_mats(self):
        """Collapses the material number so material numbers are kept as small as possible
        """

        self.mats = np.unique(self.mats, return_inverse=True)[1]  + 1

    def append(self, other_grid: "Grid"):
        return Grid(coords=np.append(self.coords, other_grid.coords, axis=0), mats=np.append(self.coords, other_grid.coords, axis=0))

    def center_and_normalize(self):
        self.center_grid_bb()
        self.scale(1 / np.amax(self.extents()))

    def copy(self):
        return Grid(coords = np.copy(self.coords), mats = np.copy(self.mats))

    def visualize(self):
        """Visualizes the array with pyvista
        """
        import pyvista as pv
        polydata = pv.PolyData(self.coords)
        polydata["Material Number"] = self.mats
        polydata.plot(eye_dome_lighting=False)


if __name__ == "__main__":
    grid = Grid(fname="example.grid")
    grid2 = grid.copy()
    grid.scale(2)
    print(grid.extents())
    print(grid2.extents())

