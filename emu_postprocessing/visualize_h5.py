"""
Extracts and plots data from an hdf5 file containing peridynamic node information.
The hdf5 (h5) fields are assumed to be formatted as numpy arrays with dimensions of
[timestep, node], with higher-dimensional data having additional array dimensions. The
available datasets and how to access them are defined in OUTPUT_DICT.

The default plot properties are defined by the dictionaries VIEWPOINT and WINDOW. These
properties are altered according to the command line arguments in apply_plot_options().

The available command line arguments are defined in PARSER_DICT. Additional command line
arguments can be created by adding to PARSER_DICT, following the existing format. The
values in OUTPUT_DICT are lists of the arguments used by the parser to activate the
options.

Requires:
    mayavi
    optparse
    os
    sys
"""
import os
import sys
from optparse import OptionParser

import h5py
from loguru import logger
import mayavi
import numpy as np
from mayavi import mlab

#########################################################################################
#                                      PARAMETERS                                       #
#########################################################################################
DEBUG = True

# fmt: off
PARSER_DICT = {
  # Label            Command line inputs      Dest           Type      Action       Default   Help
    'extents':      ['-e', '--extents',      'extents',     'string', 'store',      None,    'Spatial extents of included points [-x, +x, -y, +y, -z, +z]. Write "inf" to specify infinity.'],
    'selection':    ['-s', '--show',         'selection',   'string', 'store',      'dmg',   'Dataset selected for viewing, chosen from the labels in OUTPUT_DICT.'],
    'grid_spacing': ['-d', '--grid_spacing', 'gs',          'float',  'store',      0.5,     'Grid spacing of dataset. Sets size of datapoints.'],
    'write_image':  ['-i', '--write',        'write_image',  None,    'store_true', False,   'Whether plot will be saved to file. If so, the plot will not appear on screen.'],
    'min_plot':     ['-m', '--min_plot',     'min_plot',    'float',  'store',      None,    'Minimum displayed output value. Any datapoints below this are omitted.'],
    'max_plot':     ['-M', '--max_plot',     'max_plot',    'float',  'store',      None,    'Maximum displayed output value. Any datapoints above this are omitted.'],
    'min_legend':   ['-l', '--min_legend',   'min_legend',  'float',  'store',      None,    'Minimum color scale value. Datapoints below this are the same color.'],
    'max_legend':   ['-L', '--max_legend',   'max_legend',  'float',  'store',      None,    'Maximum color scale value. Datapoints above this are the same color.'],
    'exaggeration': ['-x', '--exag',         'exag',        'float',  'store',      0.0,     'Displacement exaggeration factor. 0.0 plots reference configuration.'],
    'timestep':     ['-t', '--timestep',     'timestep_output', 'int', 'store',     1,       'Timestep number to be viewed.'],
    'greyscale':    ['-g', '--greyscale',    'greyscale',    None,    'store_true', False,   'Whether colorscale is greyscale (`Greys`), instead of blue-red'],
    'view':         ['-v', '--view',         'view',        'string', 'store',      None,    '[unusable] View angle, one of (x+, x-, y+, y-, z+, z-, iso)'],
    'scalebar':     ['-b', '--scalebar',     'scalebar',     None,    'store_false',True,    'Whether to enable the scalebar (scalebar enabled by default)']
}

OUTPUT_DICT = {
  # Label         Output name        Extraction function
    'xloc':     ['X Coordinate',     lambda hf, t: hf['Coordinates'][t,:,0] ],
    'yloc':     ['Y Coordinate',     lambda hf, t: hf['Coordinates'][t,:,1] ],
    'zloc':     ['Z Coordinate',     lambda hf, t: hf['Coordinates'][t,:,2] ],
    'ux':       ['Displacement X',   lambda hf, t: hf['Disp'][t,:,0]     ],
    'uy':       ['Displacement Y',   lambda hf, t: hf['Disp'][t,:,1]     ],
    'uz':       ['Displacement Z',   lambda hf, t: hf['Disp'][t,:,2]     ],
    'vx':       ['Velocity X',       lambda hf, t: hf['Vel'][t,:,0]      ],
    'vy':       ['Velocity Y',       lambda hf, t: hf['Vel'][t,:,1]      ],
    'vz':       ['Velocity Z',       lambda hf, t: hf['Vel'][t,:,2]      ],
    'dmg':      ['DMG Total',        lambda hf, t: hf['DMG'][t,:]      ],
    # 'dmg':      ['DMG Total',        lambda hf, t: hf['DMG'][t,:,0]      ],
    'dmg_self': ['DMG Self',         lambda hf, t: hf['DMG'][t,:,1]      ],
    'dmg_int':  ['DMG Interface',    lambda hf, t: hf['DMG'][t,:,2]      ],
    'crit_str': ['Critical Stretch', lambda hf, t: hf['Ecrit'][t,:]      ],
    'str_max':  ['Stretch Max',      lambda hf, t: hf['Stnode'][t,:]  ],
    # 'str_max':  ['Stretch Max',      lambda hf, t: hf['Stretch'][t,:,0]  ],
    'str_min':  ['Stretch Min',      lambda hf, t: hf['Stretch'][t,:,1]  ],
    'nofail':   ['No fail value',    lambda hf, t: hf['Nofail'][t,:]     ],
    'bdry':     ['Boundary region',  lambda hf, t: hf['Nodbd'][t,:,0] ],
    'bdry1':     ['Boundary region',  lambda hf, t: hf['Nodbd'][t,:,1] ],
    'bdry2':     ['Boundary region',  lambda hf, t: hf['Nodbd'][t,:,2] ],
    # 'bdry':     ['Boundary region',  lambda hf, t: hf['Boundary'][t,:,0] ],
    'yldfr':    ['Yield Fraction',   lambda hf, t: hf['YldFr'][t,:]      ],
    'mat':      ['Material',         lambda hf, t: hf['Material'][t,:]   ],
    'proc':     ['Processor',        lambda hf, t: hf['Processor'][t,:]  ],
    'cell':     ['Cell',             lambda hf, t: hf['Cell'][t,:]       ]
}
# fmt: on

# VIEWS_DICT = {
#     'x+':   mlab.gcf().scene.x_plus_view,
#     'x-':   mlab.gcf().scene.x_minus_view,
#     'y+':   mlab.gcf().scene.y_plus_view,
#     'y-':   mlab.gcf().scene.y_minus_view,
#     'z+':   mlab.gcf().scene.z_plus_view,
#     'z-':   mlab.gcf().scene.z_minus_view,
#     'iso':  mlab.gcf().scene.isometric_view,
#     'def':  mlab.gcf().scene.isometric_view     # Default for incorrect input
# }

VIEWPOINT = {  # Viewpoint properties
    "view": None,
    "azimuth": 0,
    "elevation": 0,
    "roll": None,
    "distance": None,
    "focalpoint": (0, 0, 0),
    "parallel_projection": True,
    "parallel_scale": 3.8,
}

WINDOW = {  # Window properties
    "offscreen": False,
    "size": (1400, 600),
    "bgcolor": (1, 1, 1),
    "fgcolor": (0, 0, 0),
    "show_axes": True,
    "colormap": "blue-red",
}


#########################################################################################
#                                      FUNCTIONS                                        #
#########################################################################################
def get_filename():
    """Get h5 file name from command line argument. Assign default if not found.

    Returns:
        filename (str): name of h5 file that will be read
    """
    if len(sys.argv) < 2:
        filename = "./simulation.h5"
        print("No filename provided. Assuming ./simulation.h5 ...")
    else:
        filename = sys.argv[1]
        print("Reading", filename, "...")

    return filename


def parse_options(parser_dict):
    """Define and parse command line options according to dictionary containing option
    data.

    Args:
        parser_dict (dict): Dictionary containing necessary option parser information

    Returns:
        options (dict): Dictionary containing parsed or default option values
    """
    parser = OptionParser()

    for key in parser_dict:
        parser.add_option(
            parser_dict[key][0],
            parser_dict[key][1],
            dest=parser_dict[key][2],
            type=parser_dict[key][3],
            action=parser_dict[key][4],
            default=parser_dict[key][5],
            help=parser_dict[key][6],
        )

    options, _ = parser.parse_args()

    return options.__dict__


def options_convert_extents(options):
    """Convert extents string from input to list.
    If options has a field 'extents' and 'extents' is not nothing, convert it to a list

    Args:
        options (dict): Dictionary containing extent field to be converted (if it exists)

    Return:
        options (dict): Dictionary containing converted extent field (if it exists)
    """
    if options["extents"]:
        options["extents"] = eval(options["extents"])

    return options


def open_h5(filename):
    """Open h5 file and return root directory structure

    Args:
        filename (str): name of h5 file to be read

    Returns:
        (h5py File): h5 data structure with fields accessible [by dot notation] TODO
    """
    return h5py.File(filename, "r")


def select_output(hf, selection, timestep_index, output_dict):
    """Extract selected output quantity from h5 file.
    output_dict is used to associate selection label with full output name and lambda
    function that will return the associated data from the h5 structure.

    Args:
        hf (hdf5 file structure): h5 file opened to root directory
        selection (str): label (key) of selected output quantity
        timestep_index (int): index of timestep to be viewed
        output_dict (dict): dictionary with values containing the field name and a
                            lambda function to extract that field from the h5

    Returns:
        output (float/int nx1): list containing selected output data
    """
    print("Output=", output_dict[selection][0])
    return np.array(output_dict[selection][1](hf, timestep_index))


def get_dataset(name, data, timestep_index):
    """Extract dataset from h5 file and return as np array. Returns None if dataset not
    found in h5 file.

    Args:
        name (str): name of dataset in h5 file
        data (h5py File): h5 data structure
        timestep_index (int): index of timestep to be extracted from h5; if None, returns
                              all timesteps

    Returns:
        (array): dataset, type and dimensions dependent on case
    """
    if name in data:
        if timestep_index is not None:
            return np.array(data[name][timestep_index])
        else:
            return np.array(data[name])
    else:
        print("get_dataset: dataset not found in h5 datasets:", name)
        return None


def apply_displacements(coords, disp, exag=0.0):
    """Apply displacements to coordinates according to exaggeration.

    Args:
        coords (float, nx3 array): datapoint coordinates in [node, (x,y,z)] format
        disp (float, nx3 array): datapoint displacements in [node, (x,y,z)] format
        exag (float, optional): displacement exaggeration factor. 1.0 is true scale

    Returns:
        (float, nx3 array): datapoint coordinates after displacements
    """
    if exag != 0.0:
        return coords + disp * exag
    else:
        return coords


def cull_omitted_data(coords, output, data, timestep_index):
    """Remove dropped datapoints from dataset according to the recorded actual number of
    datapoints that is stored in the h5 file. If datapoint count is not found,

    Args:
        coords (float, nx3 array): datapoint coordinates in [node, (x,y,z)] format
        output (float, nx1 array): datapoint values corresponding with coord
        data (h5py File): h5 data structure to extract number of nodes
        timestep_index (int): index of

    Returns:
        (float, mx3 array): datapoint coordinates masked by "wanted" array
        (float, mx1 array): datapoint values masked by "wanted" array
    """
    num_nodes = get_dataset("Num_Nodes", data, timestep_index)
    if num_nodes is not None:
        if DEBUG:
            print(
                "DEBUG: cull_omitted_data: culling from", len(coords), "to", num_nodes
            )
        return coords[:num_nodes], output[:num_nodes]
    else:
        if DEBUG:
            print("DEBUG: cull_omitted_data: Num_Nodes dataset not found")
        return coords, output


def cull_data_by_location(coords, output, extents=None):
    """Trim datapoints according to specified spacial extents.
    If no extents are provided, do nothing and return inputs.

    Args:
        coords (float, nx3 array): datapoint coordinates in [node, (x,y,z)] format
        output (float, nx1 array): datapoint values corresponding with coord
        extents (float, 6x1 list, optional): spacial limits in [-x, +x, -y, +y, -z, +z]
                                             order

    Returns:
        (float, mx3 array): datapoint coordinates masked by "wanted" array
        (float, mx1 array): datapoint values masked by "wanted" array
    """
    if extents is not None:
        wanted = (
            (coords[:, 0] > extents[0])
            & (coords[:, 0] < extents[1])
            & (coords[:, 1] > extents[2])
            & (coords[:, 1] < extents[3])
            & (coords[:, 2] > extents[4])
            & (coords[:, 2] < extents[5])
        )

        return coords[wanted, :], output[wanted]
    else:
        return coords, output


def cull_data_by_value(coords, output, limits=None):
    """Trim datapoints according to specified value limits.
    If no limits are provided, do nothing and return inputs.

    Args:
        coords (float, nx3 array): datapoint coordinates in [node, (x,y,z)] format
        output (float, nx1 array): datapoint values corresponding with coord
        limits (float, 2x1 list, optional): value limits in [min, max] order

    Returns:
        (float, mx3 array): datapoint coordinates masked by "wanted" array
        (float, mx1 array): datapoint values masked by "wanted" array
    """
    if limits is not None:
        wanted = (output >= limits[0]) & (output <= limits[1])

        logger.debug(
            f"Cull by value: {np.count_nonzero(wanted)}/{len(output)} points remaining."
        )
        return coords[wanted, :], output[wanted]
    else:
        return coords, output


def set_limits(output, plot_limits=[None, None], legend_limits=[None, None]):
    """Set plot and legend limits according to inputs.
    If limits are input and valid, return them. Otherwise, set them to the according
    limit in the data. If limits are input and out of order, reverse them.

    Args:
        output (np array, nx1): output dataset being plotted
        plot_limits (float, 2x1 list, optional): [min, max] limits of values that will be
                    displayed in plot
        legend_limits (float, 2x1 list, optional): [min, max] limits of colorbar scale

    Returns:
        plot_limits (float, 2x1 list): [min, max] limits of plotted values
        legend_limits (float, 2x1 list): [min, max] limits of colorbar scale
    """
    if all(plot_limits) and plot_limits[0] > plot_limits[1]:
        plot_limits.reverse()
    if all(legend_limits) and legend_limits[0] > legend_limits[1]:
        legend_limits.reverse()

    if plot_limits[0] is None:
        plot_limits[0] = min(output)
    if plot_limits[1] is None:
        plot_limits[1] = max(output)
    if legend_limits[0] is None:
        legend_limits[0] = min(output)
    if legend_limits[1] is None:
        legend_limits[1] = max(output)

    return plot_limits, legend_limits


def apply_plot_options(options, viewpoint, window):
    """Adjust viewpoint and window properties according to input options.

    Args:
        options (dict): Dictionary containing parsed or default option values
        viewpoint (dict): dictionary containing viewpoint properties (orientation, scale)
        window (dict): dictionary containing window properties (visibility, size, color)

    Returns:
        viewpoint (dict): dictionary containing viewpoint properties after changes
        window (dict): dictionary containing window properties after changes
    """
    if options["write_image"]:
        window["offscreen"] = True
    if options["greyscale"]:
        window["colormap"] = "Greys"

    # if options['view'] is not None:
    #     if DEBUG: print('DEBUG: Storing view as', options['view'])
    #     viewpoint['view'] = VIEWS_DICT.get(options['view'], 'def')
    #     if DEBUG: print('DEBUG: View stored as', type(viewpoint['view']), viewpoint['view'])

    return viewpoint, window


def set_image_filename(h5_filename, timestep_output, selection, exag):
    """Set the filename of the saved image based on h5 name, timestep, and output
    selection.

    Args:
        h5_filename (str): name of h5 file
        timestep (int/float): timestep being viewed
        selection (str): selected output dataset
        exag (float): exaggeration factor

    Returns:
        image_filename (str): name of image to be saved
    """
    filebase, _ = os.path.splitext(h5_filename)
    primer_path = filebase.rsplit("\\", 1)[0]
    case_name = filebase.rsplit("\\", 1)[1]

    folder_name = primer_path + "\\" + selection + "_" + str(exag) + "\\"

    # If the folder does not exist then create the directory
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    image_filename = (
        folder_name + case_name + "_" + "%06d" % int(timestep_output) + ".png"
    )

    return image_filename


def plot_data(datapoints, viewpoint, window, image_filename="image.png"):
    """Plot the processed data.
    Given the data in "datapoints", plot it in the window defined by the properties in
    "window", viewing the data as defined in "viewpoint". If specified, save the plot
    with the specified (or default) filename.

    Args:
        datapoints (dict): dictionary containing the datapoint coordinates, values, and
                           scale, and the colorbar limits
        viewpoint (dict): dictionary containing viewpoint properties (orientation, scale)
        window (dict): dictionary containing window properties (visibility, size, color)
        image_filename (str, optional): name of saved plot image
    """
    mlab.options.offscreen = window["offscreen"]  # Set on- or off-screen visibility
    mlab.figure(
        size=window["size"],  # Open figure with size and colors
        bgcolor=window["bgcolor"],  #
        fgcolor=window["fgcolor"],
    )  #

    mlab.gcf().scene.parallel_projection = viewpoint[
        "parallel_projection"
    ]  # Set parallel projection state

    high = mlab.points3d(
        datapoints["x"],
        datapoints["y"],
        datapoints["z"],
        datapoints["value"],
        scale_factor=datapoints["scale"],
        vmin=datapoints["legend_limits"][0],
        vmax=datapoints["legend_limits"][1],
        scale_mode="none",
        mode="cube",
        reset_zoom=False,
        resolution=5,
        colormap=window["colormap"],
    )

    if viewpoint["view"] is not None:
        if DEBUG:
            print("DEBUG: Setting view by preset")
        # viewpoint['view']()
    elif viewpoint["azimuth"] is not None:
        if DEBUG:
            print("DEBUG: Setting view by angles")
        mlab.view(
            azimuth=viewpoint["azimuth"],  # Set camera location
            elevation=viewpoint["elevation"],  #
            roll=viewpoint["roll"],  #
            distance=viewpoint["distance"],  #
            focalpoint=viewpoint["focalpoint"],
        )  #

    if options["scalebar"]:
        mlab.scalarbar(orientation="vertical")  # Enable scalebar (vertical)

    mlab.gcf().scene.camera.parallel_scale = viewpoint[
        "parallel_scale"
    ]  # Set view scale

    if not window["offscreen"]:
        high.scene.show_axes = window["show_axes"]  # Set axes triad visibility
        mlab.show()
    else:
        mlab.savefig(image_filename)
        print("Saved", image_filename)
        mlab.close()


#########################################################################################
#                                      EXECUTION                                        #
#########################################################################################
# Obtain h5 filename
h5_filename = get_filename()

# Read and condition command line arguments
options = parse_options(PARSER_DICT)
options = options_convert_extents(options)

# Read data from h5 file
data = open_h5(h5_filename)

# Select target time step
timesteps = list(data["Time_Steps"][:].tolist())
timestep_index = timesteps.index(options["timestep_output"])

# Extract desired datasets
# coords = np.array(data['Coordinates'][timestep_index, :, :])
coords = get_dataset("Coordinates", data, timestep_index)
output = select_output(data, options["selection"], timestep_index, OUTPUT_DICT)

# Apply displacements
coords = apply_displacements(
    coords, get_dataset("Disp", data, timestep_index), options["exag"]
)

# Cull omitted data (if available)
coords, output = cull_omitted_data(coords, output, data, timestep_index)

# Cull data by location and value
coords, output = cull_data_by_location(coords, output, options["extents"])

plot_limits, legend_limits = set_limits(
    output,
    [options["min_plot"], options["max_plot"]],
    [options["min_legend"], options["max_legend"]],
)

coords, output = cull_data_by_value(coords, output, plot_limits)

# Organize data
datapoints = {
    "x": coords[:, 0],
    "y": coords[:, 1],
    "z": coords[:, 2],
    "value": output,
    "scale": options["gs"],
    "legend_limits": legend_limits,
}

# Set plot options
viewpoint, window = apply_plot_options(options, VIEWPOINT, WINDOW)
if options["write_image"]:
    image_filename = set_image_filename(
        h5_filename, options["timestep_output"], options["selection"], options["exag"]
    )
else:
    image_filename = None

# Plot data
plot_data(datapoints, viewpoint, window, image_filename)
