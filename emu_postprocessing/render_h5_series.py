import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from gooey import Gooey, GooeyParser
from mayavi import mlab
import pandas as pd


def set_general_plot_parameters(view):
    """
    """
    # Isometric, perspective
    if view == "iso":
        mlab.view(azimuth=-135, elevation=65)
        mlab.gcf().scene.parallel_projection = False
    # Cross-section, parallel projection
    elif view == "front":
        mlab.view(azimuth=0, elevation=90, focalpoint=(0, 0, -1))
        mlab.gcf().scene.parallel_projection = True
        mlab.gcf().scene.camera.parallel_scale = 4
    # Top view, parallel projection
    elif view == "top":
        mlab.view(azimuth=0, elevation=0)
        mlab.gcf().scene.parallel_projection = True
        mlab.gcf().scene.camera.parallel_scale = 100
    else:
        print("Missing options for given view. Using iso instead.")
        mlab.view(azimuth=-135, elevation=65)
        mlab.gcf().scene.parallel_projection = False


@Gooey(
    use_cmd_args=True, show_success_modal=False, return_to_config=True, header_height=20
)
def parse_options():
    parser = GooeyParser(usage="usage: %(prog)s somefile.h5")
    parser.set_defaults(
        grid_spacing=0.5,
        image_resolution=[800, 800],
        extents=None,
        show="dmg",
        write_image=False,
        min_plot=None,
        max_plot=None,
        min_legend=None,
        max_legend=None,
        exag=0.0,
        timestep_list="-1",
        write_all_timesteps=False,
        view="iso",
        scalebar=True,
    )

    # fmt: off
    parser.add_argument("filename", type=str, widget="FileChooser")
    parser.add_argument("-d","--grid_spacing",dest="grid_spacing", type=float)
    parser.add_argument("-r","--resolution", dest="image_resolution", type=float, nargs=2)
    parser.add_argument("-e","--extents",     dest="extents",     type=str, help='Spatial extents of included points [xmin, xmax, ymin, ymax, zmin, zmax].  Write "inf" to specify infinity..')
    parser.add_argument("-s","--show",        dest="show",        type=str)
    parser.add_argument("-i",                 dest="write_image", action="store_true")
    parser.add_argument("-m", "--min_plot",    dest="min_plot",    type=float)
    parser.add_argument("-M", "--max_plot",    dest="max_plot",    type=float)
    parser.add_argument("-l", "--min_legend",  dest="min_legend",  type=float)
    parser.add_argument("-L", "--max_legend",  dest="max_legend",  type=float)
    parser.add_argument("-x","--exag",        dest="exag",        type=float)
    parser.add_argument("-t","--timestep_list", dest="timestep_list", type=str, help="Timesteps must be entered as comma-separated list with no whitespace. e.g. 100,200,300")
    parser.add_argument("-a",                 dest="write_all_timesteps", action="store_true")
    parser.add_argument("-v","--view",        dest="view",        type=str)
    parser.add_argument("-b","--scalebar",    dest="scalebar",    action="store_false")
    parser.add_argument("-p","--print",       dest="print",       action="store_true", help="Print information about the data available in the specified file and exit.")
    # fmt: on

    args = parser.parse_args()
    return args


def select_output(hf, selection, timestep):
    """ Extract selected output quantity from h5 file.
    output_dict is used to associate selection label with full output name and lambda
    function that will return the associated data from the h5 structure.

    Args:
        hf (hdf5 file structure): h5 file opened to root directory
        selection (string): label of selected output quantity
        timestep_index (int): index of timestep to be viewed

    Returns:
        output (float/int nx1): list containing selected output data
    """
    return np.array(hf.loc[timestep, selection])


def list_h5_data(h5: pd.DataFrame) -> None:
    print(f"Max node number: {h5.index.max()[1]}")
    print(f"Available time steps: \n\t{list(h5.index.levels[0])}")
    print(f"Available data fields: \n\t{list(h5.columns)}")


def scatter_visualize_damage(h5_filename, plot_timesteps, opt):
    # Open and read the h5 file
    hf = pd.read_hdf(h5_filename, key="data", mode="r")
    list_h5_data(hf)
    if opt.print:
        return

    # get timesteps
    try:
        time_steps = hf.index.levels[0]
    except:
        time_steps = hf.index.unique()


    # write to screen all variabl
    print()
    print("Reading %s for input..." % h5_filename)
    print("    Exag= %s" % opt.exag)
    print("    Min_plot= %s" % opt.min_plot, "Max_plot= %s" % opt.max_plot)
    print("    Min_Color= %s" % opt.min_legend, "Max_Color= %s" % opt.max_legend)
    print("Plotting timesteps:", plot_timesteps)
    # find the timestep_index for slicing
    if plot_timesteps == "all":
        plot_timesteps = time_steps.tolist()
    elif len(plot_timesteps) == 3 and plot_timesteps[1] == -1:
        plot_timesteps = time_steps[
            (time_steps >= plot_timesteps[0]) & (time_steps <= plot_timesteps[2])
        ]

    time_steps = time_steps.tolist()

    for timestep in plot_timesteps:
        if timestep == -1:
            timestep_index = len(time_steps) - 1
            timestep = time_steps[-1]
        else:
            timestep_index = time_steps.index(timestep)

        ##check desired output
        output = select_output(hf, opt.show, timestep)

        print("Reading timestep= %s" % timestep, "Timestep index= %s" % timestep_index)
        print(
            "Min_Value %s= " % opt.show,
            min(output),
            "Max_Value %s=" % opt.show,
            max(output),
        )

        # Obtain hdf5 data set
        c1 = np.array(hf.loc[timestep, ("x1", "x2", "x3")])

        u = np.array(hf.loc[timestep, ("u1", "u2", "u3")])

        # set the coordinate locations
        x = c1[:, 0]
        y = c1[:, 1]
        z = c1[:, 2]

        # apply the exaggeration
        x = x + u[:, 0] * opt.exag
        y = y + u[:, 1] * opt.exag
        z = z + u[:, 2] * opt.exag

        # create wanted array wanted array
        wanted = np.ones_like(output, dtype=bool)
        if opt.min_plot:
            wanted = wanted & (output >= opt.min_plot)
        if opt.max_plot:
            wanted = wanted & (output <= opt.max_plot)
        if opt.extents is not None:
            wanted = (
                wanted
                & (x > opt.extents[0])
                & (x < opt.extents[1])
                & (y > opt.extents[2])
                & (y < opt.extents[3])
                & (z > opt.extents[4])
                & (z < opt.extents[5])
            )

        # slice arrays based on wanted
        x = x[wanted]
        y = y[wanted]
        z = z[wanted]
        output = output[wanted]

        if opt.write_image:
            # folder name should be view_output
            # if no view is specified then just output
            image_filename_base, ext = assemble_image_filename(
                opt, h5_filename, timestep
            )

            mlab.options.offscreen = True
            image_filename = image_filename_base + ext
            mlab.figure(size=opt.image_resolution, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            # fmt: off
            plot = mlab.points3d(
                x, y, z, output,
                scale_factor=opt.grid_spacing,
                scale_mode="none",
                mode="cube",
                reset_zoom=False,
                resolution=5,
                vmin=opt.min_legend,
                vmax=opt.max_legend,
            )
            # fmt: on
            if opt.scalebar:
                mlab.scalarbar()
            set_general_plot_parameters(opt.view)
            # plot.scene.show_axes = True
            mlab.savefig(image_filename)
            mlab.close()

        ### NOT SAVING IMAGE / INTERACTIVE PLOTTING
        else:
            mlab.options.offscreen = False
            mlab.figure(size=opt.image_resolution, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            # fmt: off
            plot = mlab.points3d(
                x, y, z, output,
                scale_factor=opt.grid_spacing,
                scale_mode="none",
                mode="cube",
                reset_zoom=False,
                resolution=5,
                vmin=opt.min_legend,
                vmax=opt.max_legend,
            )
            # fmt: on

            set_general_plot_parameters(opt.view)
            # plot.scene.show_axes = True
            if opt.scalebar:
                mlab.scalarbar()
            mlab.show()


def assemble_image_filename(opt, h5_filename, timestep):
    # folder name should be view_output
    # if no view is specified then just output

    if opt.min_legend is not None or opt.max_legend is not None:
        threshold_str = "_t%3.1f-%3.1f" % (
            opt.min_legend or 0.0,
            opt.max_legend or 1.0,
        )
    else:
        threshold_str = ""

    primer_path = Path(h5_filename).parent

    folder_name = str(primer_path) + "/" + opt.view + "_" + opt.show

    # check if the folder already exists.  if the folder exists the files will be overwritten
    directory_exists = os.path.exists(folder_name)

    # if the folder does not exist then create the directory
    if directory_exists == False:
        os.makedirs(folder_name)

    image_filename_base = (
        folder_name + "/simulation_" + "%06d" % int(timestep) + threshold_str
    )

    ext = ".png"
    return image_filename_base, ext
    # fmt: on


if __name__ == "__main__":
    args = parse_options()

    # Set h5 file name as the first argument
    h5_filename = args.filename

    # Prep the extents for separation.  eval() is used to convert from the parser input to a list of sorts then np.array() converst list to array
    if args.extents:
        args.extents = np.array(eval(args.extents))

    if args.timestep_list:
        timesteps = eval(args.timestep_list)
        if not isinstance(timesteps, list) and not isinstance(timesteps, tuple):
            timesteps = [timesteps]
    if args.write_all_timesteps:
        timesteps = "all"

    scatter_visualize_damage(h5_filename, timesteps, args)
