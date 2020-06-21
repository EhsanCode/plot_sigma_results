#!/usr/bin/env python

import scipy
from scipy.io import loadmat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import SaliencyLib


def get_sigma_max_score(metric_scores, metric_number):
    if metric_number != 3:
        max_score = metric_scores.max()
    else:
        max_score = metric_scores.min()
    sigma = np.where(metric_scores == max_score)[0][0] + 1
    return sigma, max_score


def plot_save_metric_scores(metric_scores, output_path, metric_number):
    sigma, max_score = get_sigma_max_score(metric_scores, metric_number)

    if metric_number < 3:
        min_score_lim = 0.5
        max_score_lim = 1.01
        major_step = 0.1
        minor_step = 0.01
    elif metric_number == 3:
        min_score_lim = 0
        max_score_lim = 31
        major_step = 10
        minor_step = 1
    else:
        min_score_lim = 0
        max_score_lim = 2.6
        major_step = 0.5
        minor_step = 0.1

    fig, ax = plt.subplots()
    ax.plot(range(1, 1 + len(metric_scores)), metric_scores)
    ax.plot(sigma, max_score, 'or', alpha=0.4)
    ax.plot([sigma, sigma], [min_score_lim + 0, max_score], 'r', alpha=0.4)
    plt.text(sigma + 8, (min_score_lim + max_score) / 2, r'$\sigma$=%d' %(sigma), fontsize=10, rotation=90, rotation_mode='anchor', color='g')
    ax.plot([1, sigma], [max_score, max_score], 'r', alpha=0.4)
    plt.text(1, max_score + max_score * 0.01, 'Max Score=%5.2f' %(max_score), fontsize=7, rotation=0, rotation_mode='anchor', color='g')

    ax.set(xlabel='Sigma', ylabel='Similarity Score')

    plt.xlim(1, len(metric_scores))
    plt.ylim(min_score_lim, max_score_lim)
    major_ticks_x = np.arange(0, len(metric_scores) + 0.1, 50)
    minor_ticks_x = np.arange(0, len(metric_scores) + 0.1, 5)
    major_ticks_y = np.arange(min_score_lim, max_score_lim, major_step)
    minor_ticks_y = np.arange(min_score_lim, max_score_lim, minor_step)
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
    file_path = 'RANDOM_360x640_NP_100_d_20_NC_2.mat'
    mat = loadmat(file_path)
    mat_data = mat['results']
    scores = mat_data['scores'][0, 0][0]
    gaze_image = mat_data['imageInfo'][0, 0][0, 0]['image']
    for metric_number in range(5):
        output_path = 'graph_' + str(metric_number) + '.pdf'
        metric_scores = scores[:, metric_number]
        plot_save_metric_scores(metric_scores, output_path, metric_number)

        output_path = 'saliency_map_' + str(metric_number) + '.pdf'
        sigma = get_sigma_max_score(metric_scores, metric_number)[0]
        saliency_map = SaliencyLib.get_saliency_map(gaze_image, sigma)
        SaliencyLib.show_save_image(saliency_map, output_path)


if __name__ == '__main__':

    main()
