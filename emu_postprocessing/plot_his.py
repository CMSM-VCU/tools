""" Extracts, organizes, and plots data from an Emu history file.
History file must be in column-wise format with each column being exactly 15 characters
wide. The first row of the history file may contain anything, but must end with the
number of node histories preceded by whitespace. The second row of the history file must
be the column titles.

The history file name (and path) is expected as the first command-line argument. If it is
omitted, the file name is assumed to be "./emu.his".
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


#########################################################################################
#                                      PARAMETERS                                       #
#########################################################################################
COLUMN_WIDTH = 15

NAME_MAP = {
    'iter':      'Timestep',
    'timex':     'Time',
    'dt':        'Step size',
    'phid':      'Dissipated energy',
    'ekin':      'Kinetic energy',
    'work_bdry': 'Boundary work'
}

CATEGORIES = ('x1',  'x2',    'x3',
              'u1',  'u2',    'u3',
              'v1',  'v2',    'v3',
              'dmg', 'stmax', 'scmax',
              'force')


#########################################################################################
#                                      FUNCTIONS                                        #
#########################################################################################
def get_filename():
    """ Get history file name from command line argument. Assign default if not found.

    Returns:
        filename (str): name of history file that will be read
    """
    if len(sys.argv) < 2:
        filename = './emu.his'
        print('No filename provided. Assuming ./emu.his ...')
    else:
        filename = sys.argv[1]
        print('Reading', filename, '...')

    return filename


def get_number_of_node_histories(filename):
    """ Get number of node histories in history file from first line of history file.
    Requires that last item on first line of history file is the number of node
    histories, separated by whitespace.

    Args:
        filename (str): name of history file

    Returns:
        num_histories (int): number of node histories
    """
    with open(filename) as file:
        first_line = file.readline()
        num_histories = int(first_line.split()[-1])
    print('Reading from %i node histories' % (num_histories))
    return num_histories


def read_his_data(filename):
    """ Parse data from history file.
    Requires that the second line of the history file contains the column names and that
    each column is exactly 15 characters wide. There is no requirement for number of
    columns or rows.

    Args:
        filename (str): name of history file

    Returns:
        (structured np array): all extracted data with column names. Number of rows is
                               equal to the number of timesteps or history dumps
    """
    return np.genfromtxt(filename, dtype=None, delimiter=COLUMN_WIDTH,
                         names=True, skip_header=1)


def remap_column_names(data_names, name_map):
    """ Remap data array column names using dictionary map.
    For each column name that matches a key in name_map, the column name is replaced with
    that key's value.

    Args:
        data_names (str, nx1 tuple): list of column names taken from structured np array
        name_map (dict): dictionary with keys matching history file column names to be
                         replaced by the corresponding values

    Returns:
        (str, nx1 tuple): New list of column names
    """
    return tuple(name_map.get(name, name) for name in data_names)
    # get(name, name) means it will keep the current name if not found in dictionary


def populate_datasets(categories, data_names):
    """ Populate dictionary of dataset labels with the corresponding column names.
    Creates a dictionary with keys from the list "categories". For each key, if a column
    name contains that key, the name is added to a tuple of strings stored under that
    key. This is used to access a set of data columns using the names stored under each
    dataset key. Intended to gather node history datasets, which should have columns
    names in the form u1_001, u1_002, u1_003, ... to be stored under the dataset
    label "u1"

    Args:
        categories (str, nx1 tuple): list of dataset labels used to locate corresponding
                                     column names
        data_names (str, mx1 tuple): list of column names taken from structured np array

    Returns:
        datasets (dict): dictionary containing sets of column names
    """
    datasets = {}
    for key in categories:
        datasets.update({key: tuple(name for name in data_names if key in name)})
    return datasets


#########################################################################################
#                                      EXECUTION                                        #
#########################################################################################
filename = get_filename()

num_histories = get_number_of_node_histories(filename)

data = read_his_data(filename)

print('Reading from %i node histories' % (num_histories))
print(data.dtype.names)

data.dtype.names = remap_column_names(data.dtype.names, NAME_MAP)

datasets = populate_datasets(CATEGORIES, data.dtype.names)
# For example, the following line gets u1 for all node histories (u1_001, u1_002, ...)
#     for label in datasets['u1']: data[label]

# Add dataset that contains all columns
datasets.update({'all': data.dtype.names})

# Separate time and timestep for easy access
time = data['Time']
timestep = data['Timestep']

""" Plot data """
plotdata = []
for name in datasets['all']:
    trace = go.Scatter(
        x = time,
        y = data[name],
        mode = 'lines',
        name = name
    )
    plotdata.append(trace)

layout = go.Layout(
    autosize=False,
    width=800,
    height=600,
    margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=go.layout.XAxis(
        title='Time (s)',
        titlefont=dict(size=30),
        automargin=True
    ),
    paper_bgcolor='#7f7f7f',
    plot_bgcolor='#c7c7c7'
)

fig = go.Figure(data=plotdata, layout=layout)
pyoff.plot(fig)
