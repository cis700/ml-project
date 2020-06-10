from matplotlib import pyplot as plt
from os.path import join


class Plot:

    def __init__(self):
        self.plot_file_path = '../../data/plots/data_visualization/'
        self.dataset_name = 'CICIDS2017'

    def show_histogram(self, data, display=False):
        plt.title("{} - Histogram".format(self.dataset_name))
        plt.savefig(join(self.plot_file_path, 'histogram.png'))
        if display:
            plt.show()
