#!/usr/bin/env python
import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def read_graph_data(file_path, graph_number):
    mat_data = loadmat(file_path)
    scores = mat_data['results']['scores'][0, 0][0]
    graph_data = scores[:, graph_number]
    return graph_data


def plot_save_graph_data(graph_data, output_path, graph_number):
    fig, ax = plt.subplots()
    ax.plot(range(1, 1 + len(graph_data)), graph_data)

    ax.set(xlabel='Sigma', ylabel='Similarity Score')
    plt.xlim(0, )

    major_ticks_x = np.arange(0, len(graph_data) + 0.1, 50)
    minor_ticks_x = np.arange(0, len(graph_data) + 0.1, 5)

    if graph_number < 3:
        major_ticks_y = np.arange(0.5, 1.09, 0.1)
        minor_ticks_y = np.arange(0.5, 1.009, 0.01)
    elif graph_number == 3:
        major_ticks_y = np.arange(0, 31, 10)
        minor_ticks_y = np.arange(0, 31, 1)
    else:
        major_ticks_y = np.arange(0, 2.6, 0.5)
        minor_ticks_y = np.arange(0, 2.6, 0.1)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    ax.grid(which='both')

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    fig.savefig(output_path)
    plt.show()


def main():
    for graph_number in range(5):
        file_path = 'RANDOM_360x640_NP_100_d_20_NC_2.mat'
        output_path = 'graph_' + str(graph_number) + '.pdf'
        graph_data = read_graph_data(file_path, graph_number)
        plot_save_graph_data(graph_data, output_path, graph_number)


if __name__ == '__main__':

    main()
