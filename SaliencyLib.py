
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt

def _get_gaussian_distribution(sigma, s):
    if sigma <= 1:
        gaussian_distribution = np.ones([1, 1])
    else:
        half_width = 3 * sigma
        grid_numbers = np.array(range(-half_width, half_width + 1))
        xx, yy = np.meshgrid(grid_numbers, grid_numbers)
        gaussian_distribution = np.exp(-1/(2*sigma ^ 2)
                                    * (np.power(xx, 2) + np.power(yy, 2)))
        gs = gaussian_distribution.shape
        gaussian_distribution = gaussian_distribution[(gs[0] // 2 - s[0] // 2) : (gs[0] // 2 + s[0] // 2), 
                                                      (gs[1] // 2 - s[1] // 2) : (gs[1] // 2 + s[1] // 2)]
    
    return gaussian_distribution


def get_saliency_map(gaze_image, sigma):
    gaussian_distribution = _get_gaussian_distribution(sigma, gaze_image.shape)
    saliency_map = signal.convolve2d(
        gaze_image, gaussian_distribution, boundary='symm', mode='same')
    saliency_map /= saliency_map.max()
    return saliency_map


def get_heatmap(saliency_map):
    saliency_shape = saliency_map.shape
    heat_map = np.zeros([saliency_shape[0], saliency_shape[1], 3]) 
    heat_map[:,:,0] = saliency_map
    heat_map[:,:,0] = (heat_map[:,:,0] - 0.25) / 0.75 
    heat_map[:,:,0] = np.where(heat_map[:,:,0] < 0 , 0, heat_map[:,:,0])

    heat_map[:,:,1] = saliency_map
    # heat_map[:,:,2] = saliency_map
    heat_map[:,:,1] = np.where(heat_map[:,:,1] > 0.5 , 1 - heat_map[:,:,1], heat_map[:,:,1])
    # heat_map[:,:,1] *= 2
    # heat_map[:,:,2] = np.where(heat_map[:,:,2] > 0.75 , 1 - heat_map[:,:,2], heat_map[:,:,2])


    return heat_map

def apply_heatmap(image, saliency_map):
    heat_map = get_heatmap(saliency_map)
    image_heatmap = np.copy(image)
    image_heatmap[:,:,0] = image_heatmap[:,:,0] * (1 - saliency_map) + heat_map[:,:,0] * saliency_map
    image_heatmap[:,:,1] = image_heatmap[:,:,1] * (1 - saliency_map) + heat_map[:,:,1] * saliency_map
    image_heatmap[:,:,2] = image_heatmap[:,:,2] * (1 - saliency_map) + heat_map[:,:,2] * saliency_map
    image_heatmap = np.where(image_heatmap > 1, 1, image_heatmap)
    return image_heatmap

def show_save_image(img, output_path):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    fig.savefig(output_path)
    # plt.show()
