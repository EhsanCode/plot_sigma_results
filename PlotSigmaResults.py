#!/usr/bin/env python

from scipy.io import loadmat

mat_path = 'D:/program/Share_BTH/sigma/results_output/randomFixLast_fix_d/'


def read_graph_data(file_name):
    mat_data = loadmat(mat_path + file_name)
    return graph_data


def main():
    file_name = 'RANDOM_720x1280_NP_100_d_30_NC_2.mat'
    grhaph_data = read_graph_data(file_name)


if __name__ == '__main__':
    
    main()
