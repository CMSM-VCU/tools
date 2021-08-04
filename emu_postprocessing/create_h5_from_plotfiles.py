""" Extracts and compresses data from Emu plot files.
Plot files must be in CSV format. The first row of each plot file must be the column titles.

Plot files must be name [filename].[timestep].[processor]

The h5 fields will be named according to the dictionary "h5_labels". The dictionary keys
are the column names in the plot files and the dictionary values are what they will be
saved as in the h5. Any column names not found in "h5_labels" will be saved as is in the
h5.

Vector quantities can be specified in the "vectors" tuple and will be saved to the h5 as
vectors. To do this, the values must be saved in the plot files in columns titled
[name]1, [name]2, [name]3. These columns will then be converted to a single vector field
of name [name]. The length of the vector is assumed to be 3, but can be changed with the
optional "isize" argument.

Matrix quantities can be specified in the "matrices" tuple and will be saved to the h5 as
matrices. To do this, the values must be saved in the plot files in columns titled
[name]11, [name]12, ... [name]32, [name]33. These columns will then be converted to a
single matrix field of name [name]. The size of the matrix is assumed to be 3x3, but can
be changed with the optional "isize" and "jsize" arguments. The matrices must be 2D.

Special vector cases can be specified in the "special" dictionary. The dictionary key
will be the name of the resulting vector field, and the dictionary value is the list of
column names that will make up the vector. Special vector cases can be of any length, but
must be 1D.

The list "exclude" can be used to exclude fields from the output h5 for space-saving.
The list uses the original column names in the plot files.
"""
import numpy as np
import h5py
import glob
import os

#########################################################################################
#                                      PARAMETERS                                       #
#########################################################################################
# fmt: off
H5_LABELS = {
    "iter":         "Time_Steps",
    "node_type":    "Material",
    "nofail":       "Nofail",
    "mypr":         "Processor",
    "nodbd":        "Boundary",
    "x":            "Coordinates",
    "u":            "Disp",
    "v":            "Vel",
    "damage":       "DMG",
    "ecnode":       "Ecrit",
    "stretch":      "Stretch",
    "yldfr":        "Yield_Frac",
    "wt":           "W",
    "fnorm":        "Fnorm_Fac",
    "edt":          "Diss_Energy_Dens",
    "von_mises":    "Von_Mises",
    "timex":        "Time",
    "strs_cauchy":  "Cauchy_Stress",
    "left_strtch":  "Left_Stretch",
    "num_nodes":    "Num_Nodes"
}

VECTORS = ("x", "u", "v", "nodbd")
MATRICES = ("strs_cauchy", "left_strtch")
SPECIAL = {
    "damage": ["dmg", "dmgi", "dmgm"],
    "stretch": ["stnode", "scnode"]
}
EXCLUDE = np.hstack([[name+str(i) for name in VECTORS for i in range(1, 4)],
                     [name+str(i)+str(j) for name in MATRICES for i in range(1, 4)
                                                        for j in range(1, 4)],
                     "dmg", "dmgi", "dmgm", "stnode", "scnode", "m_global"])
# fmt: on

#########################################################################################
#                                      FUNCTIONS                                        #
#########################################################################################
def read_individual_file(filename):
    """ Parse data from a single plot file
    Returns structured Numpy array with column titles as given in plot file

    Args:
        filename (str): full filename of file to be read

    Returns:
        (structured np array): all extracted data with column titles read from first line
                               of file. Dimensions [nodes x number of fields]
    """
    try:
        data = np.genfromtxt(filename, dtype=None, delimiter=",", names=True)
        if not data.shape:
            data = np.squeeze(np.reshape(data, (1, -1)), axis=-1)
        return data
    except ValueError:
        print("Error reading file (probably inconsistent number of columns):", filename)
    except:
        print("Something went wrong:", filename)


def read_timestep_files(file_list):
    """
    """
    return np.concatenate([read_individual_file(file) for file in not_empty])


def convert_columns_to_vector(name_base, data, isize=3):
    """ Convert series of columns from plot file data into single vector field. Columns
    are selected based on the naming convention [name_base][i], i=1:isize.

    Args:
        name_base (str): name of columns (keys) to be used in vector, excluding indices
        data (dict): dictionary containing datasets under keys specified by name_base
        isize (int, optional): length of vector to be created, corresponding with number
                               of columns to be read

    Returns:
        (np array): resulting vector data for all timesteps and nodes.
                    Dimensions [timesteps x nodes x isize]
    """
    if any(name_base in key for key in data.keys()):
        return np.stack(
            [data[key] for key in [name_base + str(i) for i in range(1, isize + 1)]],
            axis=-1,
        )
    else:
        print("Vector: Name not found in columns:", name_base)
        return None


def convert_columns_to_matrix(name_base, data, isize=3, jsize=3):
    """ Convert series of columns from plot file data into single matrix field. Columns
    are selected based on the naming convention [name_base][i][j], i=1:isize, j=1:jsize.

    Args:
        name_base (str): name of columns (keys) to be used in matrix, excluding indices
        data (dict): dictionary containing datasets under keys specified by name_base
        isize (int, optional): number of rows in matrix data to be created
        jsize (int, optional): number of columns in matrix data to be created. Number of
                               columns to be read from data is (isize)x(jsize)

    Returns:
        (np array): resulting matrix data for all timesteps and nodes.
                    Dimensions [timesteps x nodes x isize x jsize]
    """
    # fmt: off
    if any(name_base in key for key in data.keys()):
        return np.stack([np.stack([data[key] for key in
                                  [name_base+str(i)+str(j) for i in range(1, isize+1)]], axis=-1)
                                                     for j in range(1, jsize+1)], axis=-1)
    # fmt: on
    else:
        print("Matrix: Name not found in columns", name_base)
        return None


def convert_columns_to_special(columns, data):
    """ Convert series of columns from plot file data into single vector field. Columns
    are selected according to the column titles in "columns".

    Args:
        columns (str, nx1 list): column titles (dictionary keys) to be used in vector
        data (dict): dictionary containing datasets under keys specified in "columns"

    Returns:
        (np array): resulting vector data for all timesteps and nodes.
                    Dimensions [timesteps x nodes x len(columns)]
    """
    # TODO: Check if all special columns are present, or omit columns that are not
    # print(type(any(columns[0] in key for key in data.keys())))
    # if all([any(name in key for key in data.keys) for name in columns]):
    if True:
        return np.stack([data[key] for key in columns], axis=-1)
    else:
        print("Special: Not all special columns found.")
        return None


def parse_file_list():
    """ Extract the filename template and available timesteps from the files in the
    current directory. This assumes that the plot files are named in the format
    [filename_base].[timestep].[processor]

    Returns:
        filename_base (str): filename template excluding timestep and processor values
        timesteps (int, nx1 list): list of all timesteps covered by plot files
    """
    # Assuming filename format of [filename_base].[timestep].[processor]
    # Extract list of files from processor 0
    file_list = glob.glob("*.0")

    # Get filename base by stripping off timestep and processor
    filename_base = file_list[0].rsplit(".", 2)[0]

    # With these files, extract all available timesteps
    timesteps = []
    for file in file_list:
        timesteps.append(int(file.split(".")[-2]))

    timesteps = sorted(timesteps)

    return filename_base, timesteps


def check_if_empty(file):
    """
    """
    with open(file) as f:
        for i, l in enumerate(f):
            pass
        if i > 0:
            is_empty = False
        else:
            is_empty = True

    return is_empty


#########################################################################################
#                                      EXECUTION                                        #
#########################################################################################
filename_base, timesteps = parse_file_list()

output_data = {}

num_nodes = []
num_fields = None

# Extract raw data and store in output_data dictionary
for i, ts in enumerate(timesteps):
    filename = filename_base + "." + str(ts) + ".*"
    file_list = glob.glob(filename)

    not_empty = []
    for file in file_list:
        if not check_if_empty(file):
            not_empty.append(file)

    if not_empty:
        input_data = read_timestep_files(not_empty)
        print(ts, ": ", input_data.shape, len(input_data.dtype.names))
        num_nodes.append(input_data.shape[0])

        if not num_fields:
            num_fields = len(input_data.dtype.names)

        if input_data.shape[0] < max(num_nodes):
            print(input_data.shape[0], "<", max(num_nodes), "padding...")
            input_data.resize(max(num_nodes))

        if i == 0:  # Initialize dictionary entries
            for name in input_data.dtype.names:
                output_data.update({name: input_data[name]})
        else:  # Append subsequent timesteps to existing dictionary entries
            for name in input_data.dtype.names:
                output_data[name] = np.dstack((output_data[name], input_data[name]))
    else:
        print("Warning: Timestep", ts, "contains no data.")

# Reorder dimensions in output_data arrays and remove extra dimensions
for name in output_data:
    output_data[name] = np.squeeze(np.transpose(output_data[name], (2, 1, 0)))

# Process vector quantities
for name in VECTORS:
    output_data.update({name: convert_columns_to_vector(name, output_data)})

# Process matrix quantities
for name in MATRICES:
    output_data.update({name: convert_columns_to_matrix(name, output_data)})

# Process special quantities
for name, columns in SPECIAL.items():
    output_data.update({name: convert_columns_to_special(columns, output_data)})

# Remove excluded data
for key in EXCLUDE:
    del output_data[key]

# Add numbers of nodes
output_data.update({"num_nodes": np.asarray(num_nodes)})

# Remove extra dimension from time and timestep fields
# Dimensions from [timesteps x nodes] to [timesteps]
output_data["timex"] = output_data["timex"][:, 0]
output_data["iter"] = output_data["iter"][:, 0]

# Open h5 file
hf = h5py.File("simulation.h5", "w")

# Save output_data to h5 file
for name in output_data:
    print(output_data[name].shape)
    hf.create_dataset(
        name=H5_LABELS.get(name, name),
        data=output_data[name],
        compression="gzip",
        compression_opts=9,
    )

hf.close()
print(
    "Saved",
    output_data["mypr"].shape[1],
    "nodes across",
    output_data["mypr"].shape[0],
    "timesteps to simulation.h5",
)
