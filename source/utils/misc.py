import os

import numpy as np

from datetime import datetime


def init_outfile(nvars):
    now = datetime.now()
    now_to_string = now.strftime("%Y_%m_%d@%H_%M_%S")
    root_dir = '../../outputs/'
    filename = 'MH_{}.csv'.format(now_to_string)
    path = os.path.join(root_dir, filename)
    with open(path, 'w') as outfile:
        outfile.write(', '.join(['parameter_{}'.format(i) for i in range(nvars - 1)] + ['variance']) + '\n')
    return path


def update_outfile(path, z, var):
    with open(path, 'a') as outfile:
        z_str = [str(z_) for z_ in z] if type(z) is np.ndarray else [str(z)]
        outfile.write(', '.join(z_str + [str(var)]) + '\n')
