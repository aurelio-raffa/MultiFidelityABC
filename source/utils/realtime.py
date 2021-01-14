import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.animation import FuncAnimation


if __name__ == '__main__':

    list_of_files = glob.glob('../../outputs/*.csv')
    latest_file = max(list_of_files, key=os.path.getctime)

    def animation(i):
        plt.style.use('ggplot')
        data = pd.read_csv(latest_file)
        plt.clf()
        for i, var in enumerate(data.columns):
            ax = plt.subplot(data.shape[1], 1, i+1)
            ax.plot(data[var])

    animate = FuncAnimation(plt.gcf(), animation, interval=250)
    plt.show()
