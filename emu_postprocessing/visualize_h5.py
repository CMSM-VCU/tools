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
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from gooey import Gooey, GooeyParser
from mayavi import mlab

#########################################################################################
#                                      PARAMETERS                                       #
#########################################################################################
DEBUG = True

# fmt: off
PARSER_DICT = {
  # Label            Command line inputs      Dest           Type      Action       Default   Help
    "filename":     ["filename", None,        None,          str,    "store",      "simulation.h5", "Path and name of the h5 file to be visualized."],
    "extents":      ["-e", "--extents",      "extents",      str,    "store",      None,    "Spatial extents of included points [-x, +x, -y, +y, -z, +z]. Write 'inf' to specify infinity."],
    "selection":    ["-s", "--show",         "selection",    str,    "store",      "dmg",   "Dataset selected for viewing, chosen from the labels in OUTPUT_DICT."],
    "grid_spacing": ["-d", "--grid_spacing", "gs",           float,  "store",      0.5,     "Grid spacing of dataset. Sets size of datapoints."],
    "write_image":  ["-i", "--write",        "write_image",  None,   "store_true", None,   "Whether plot will be saved to file. If so, the plot will not appear on screen."],
    "min_plot":     ["-m", "--min_plot",     "min_plot",     float,  "store",      None,    "Minimum displayed output value. Any datapoints below this are omitted."],
    "max_plot":     ["-M", "--max_plot",     "max_plot",     float,  "store",      None,    "Maximum displayed output value. Any datapoints above this are omitted."],
    "min_legend":   ["-l", "--min_legend",   "min_legend",   float,  "store",      None,    "Minimum color scale value. Datapoints below this are the same color."],
    "max_legend":   ["-L", "--max_legend",   "max_legend",   float,  "store",      None,    "Maximum color scale value. Datapoints above this are the same color."],
    "exaggeration": ["-x", "--exag",         "exag",         float,  "store",      0.0,     "Displacement exaggeration factor. 0.0 plots reference configuration."],
    "timestep":     ["-t", "--timestep",     "timestep_output", int, "store",      1,       "Timestep number to be viewed."],
    "greyscale":    ["-g", "--greyscale",    "greyscale",    None,   "store_true", None,   "Whether colorscale is greyscale (`Greys`), instead of blue-red"],
    "view":         ["-v", "--view",         "view",         str,    "store",      None,    "[unusable] View angle, one of (x+, x-, y+, y-, z+, z-, iso)"],
    "scalebar":     ["-b", "--scalebar",     "scalebar",     None,   "store_false",None,    "Whether to enable the scalebar (scalebar enabled by default)"],
    "list":         ["-p", "--print",        "print",        None,   "store_true", None,    "Print information about the data available in the specified file and exit."]
}

OUTPUT_DICT = {
  # Label         Output name        Extraction function
    "xloc":     ["X Coordinate",     lambda hf, t: hf["Coordinates"][t,:,0] ],
    "yloc":     ["Y Coordinate",     lambda hf, t: hf["Coordinates"][t,:,1] ],
    "zloc":     ["Z Coordinate",     lambda hf, t: hf["Coordinates"][t,:,2] ],
    "ux":       ["Displacement X",   lambda hf, t: hf["Disp"][t,:,0]     ],
    "uy":       ["Displacement Y",   lambda hf, t: hf["Disp"][t,:,1]     ],
    "uz":       ["Displacement Z",   lambda hf, t: hf["Disp"][t,:,2]     ],
    "vx":       ["Velocity X",       lambda hf, t: hf["Vel"][t,:,0]      ],
    "vy":       ["Velocity Y",       lambda hf, t: hf["Vel"][t,:,1]      ],
    "vz":       ["Velocity Z",       lambda hf, t: hf["Vel"][t,:,2]      ],
    "dmg":      ["DMG Total",        lambda hf, t: hf["DMG"][t,:]      ],
    # "dmg":      ["DMG Total",        lambda hf, t: hf["DMG"][t,:,0]      ],
    "dmg_self": ["DMG Self",         lambda hf, t: hf["DMG"][t,:,1]      ],
    "dmg_int":  ["DMG Interface",    lambda hf, t: hf["DMG"][t,:,2]      ],
    "crit_str": ["Critical Stretch", lambda hf, t: hf["Ecrit"][t,:]      ],
    "str_max":  ["Stretch Max",      lambda hf, t: hf["Stnode"][t,:]  ],
    # "str_max":  ["Stretch Max",      lambda hf, t: hf["Stretch"][t,:,0]  ],
    "str_min":  ["Stretch Min",      lambda hf, t: hf["Stretch"][t,:,1]  ],
    "nofail":   ["No fail value",    lambda hf, t: hf["Nofail"][t,:]     ],
    "bdry":     ["Boundary region",  lambda hf, t: hf["Nodbd"][t,:,0] ],
    "bdry1":     ["Boundary region",  lambda hf, t: hf["Nodbd"][t,:,1] ],
    "bdry2":     ["Boundary region",  lambda hf, t: hf["Nodbd"][t,:,2] ],
    # "bdry":     ["Boundary region",  lambda hf, t: hf["Boundary"][t,:,0] ],
    "yldfr":    ["Yield Fraction",   lambda hf, t: hf["YldFr"][t,:]      ],
    "mat":      ["Material",         lambda hf, t: hf["Material"][t,:]   ],
    "proc":     ["Processor",        lambda hf, t: hf["Processor"][t,:]  ],
    "cell":     ["Cell",             lambda hf, t: hf["Cell"][t,:]       ]
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
@Gooey(use_cmd_args=True, show_success_modal=False, header_height=20)
def parse_options(parser_dict):
    """Define and parse command line options according to dictionary containing option
    data.

    Args:
        parser_dict (dict): Dictionary containing necessary option parser information

    Returns:
        options (dict): Dictionary containing parsed or default option values
    """
    parser = GooeyParser()

    for option in parser_dict.values():
        if not option[0].startswith("-"):
            parser.add_argument(
                option[0],
                type=option[3],
                action=option[4],
                default=option[5],
                help=option[6],
                widget="FileChooser",
            )
        elif option[4] not in ["store_true", "store_false"]:
            parser.add_argument(
                option[0],
                option[1],
                dest=option[2],
                type=option[3],
                action=option[4],
                default=option[5],
                help=option[6],
            )
        else:
            parser.add_argument(
                option[0], option[1], dest=option[2], action=option[4], help=option[6],
            )

    args = parser.parse_args()

    return args.__dict__


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
    else:
        options["extents"] = [-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]

    return options


def open_h5(filename):
    """Open h5 file and return root directory structure

    Args:
        filename (str): name of h5 file to be read

    Returns:
        (h5py File): h5 data structure with fields accessible [by dot notation] TODO
    """
    return pd.read_hdf(filename, key="data", mode="r")


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
    h5path = Path(h5_filename)

    primer_path = h5path.parent
    case_name = h5path.stem

    folder_name = primer_path / f"{selection}_{exag}"
    folder_name.mkdir(exist_ok=True)

    return folder_name / f"{case_name}_{timestep_output:06d}.png"


def plot_data(datapoints, plot_options, viewpoint, window, image_filename="image.png"):
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
        datapoints["x1"] + datapoints["u1"] * plot_options["exag"],
        datapoints["x2"] + datapoints["u2"] * plot_options["exag"],
        datapoints["x3"] + datapoints["u3"] * plot_options["exag"],
        datapoints[plot_options["data_name"]],
        scale_factor=plot_options["scale"],
        vmin=plot_options["legend_limits"][0],
        vmax=plot_options["legend_limits"][1],
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


def list_h5_data(h5: pd.DataFrame) -> None:
    print(f"Max node number: {h5.index.max()[1]}")
    print(f"Available time steps: \n\t{list(h5.index.levels[0])}")
    print(f"Available data fields: \n\t{list(h5.columns)}")


#########################################################################################
#                                      EXECUTION                                        #
#########################################################################################
# Read and condition command line arguments
options = parse_options(PARSER_DICT)
options = options_convert_extents(options)
h5_filename = options["filename"]

# Read data from h5 file
data = open_h5(h5_filename)

if options["print"]:
    list_h5_data(data)
    exit(0)

# Extract desired datasets
coords = data.loc[options["timestep_output"], ("x1", "x2", "x3")]
disp = data.loc[options["timestep_output"], ("u1", "u2", "u3")]
output = data.loc[options["timestep_output"], options["selection"]]

extents = options["extents"]
extent_mask = (
    (coords["x1"] >= extents[0])
    & (coords["x1"] <= extents[1])
    & (coords["x2"] >= extents[2])
    & (coords["x2"] <= extents[3])
    & (coords["x3"] >= extents[4])
    & (coords["x3"] <= extents[5])
)

plot_limits, legend_limits = set_limits(
    output,
    [options["min_plot"], options["max_plot"]],
    [options["min_legend"], options["max_legend"]],
)

value_mask = (output >= plot_limits[0]) & (output <= plot_limits[1])

datapoints = pd.concat([coords, disp, output], axis=1)
datapoints = datapoints[extent_mask & value_mask]

# Organize data
plot_options = {
    "scale": options["gs"],
    "legend_limits": legend_limits,
    "exag": options["exag"],
    "data_name": options["selection"],
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
plot_data(datapoints, plot_options, viewpoint, window, image_filename)
