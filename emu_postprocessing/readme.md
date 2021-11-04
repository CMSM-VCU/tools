## Tools for remote system

On the remote system, where simulations are executed, the h5 file needs to be generated from the plotfiles. This makes use of the `environment_h5gen.yml` Conda environment and the `create_h5_from_plotfiles.py` script.

Create and activate the Conda environment with

```bash
$ conda env create -f environment_h5gen.yml
$ conda activate emu-h5
```

The h5 creation script can be used with

```bash
(emu-h5) $ python create_h5_from_plotfiles.py [DIRECTORY]
```

The path to a directory containing plotfiles can be specified, e.g. `./project/case/plotfiles/` The script will look for plotfiles and save the resulting h5 file in the specified directory. If no directory is specified, the script will use the current directory.

Note that the location of the script file does not matter.

## Tools for local system

The `environment_viz.yml` Conda environment and the rest of the scripts are to be used on your local computer.

### `render_h5_series.py`

This is the primary batch visualization script. Running the script with no arguments will open a GUI for selecting options. Use the `--ignore-gooey` argument to skip the GUI and use only the command line arguments. The `print` option will show you what datasets and timesteps are available in the h5 file.

### `grid_to_h5.py`

Use this script to convert an Emu grid file to an h5 file compatible with the rest of the postprocessing tools.

```bash
(emu-post) $ python grid_to_h5.py FILE
```
