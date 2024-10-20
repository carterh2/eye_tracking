"""
nemo_eyetracking
Copyright (C) 2022 Utrecht University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pathlib as pathlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

# ------------------------------------------------------------------------------
# Convenience functions for easier loading of images/saliency maps and creating
# the ground truth representations from fixation data


def load_smap(path, gray=True):
    """Convenience function to quickly load an image (saliency map) into a numpy
    array in the right format.

    Args:
        path: string or Path object for the path to the image
        gray (bool, optional): Whether to conver the image to grayscale. Defaults to True.

    Returns:
        smap: numpy array of black-white image from input path
    """
    smap = Image.open(path)  # load image

    # convert to grayscale if desired, otherwise retain color channels
    if gray:
        smap = smap.convert("L")
    else:
        smap = smap.convert("RGB")
        
    smap = smap.transpose(Image.FLIP_TOP_BOTTOM)

    smap = np.asarray(smap)  # convert to numpy array
    return smap


def make_gt_smap(
    x, y, img_dims=(1080, 1920), blur=False, blur_factor=(44.88, 44.88), fix_value=255
):
    """Creates a 2D ground truth saliency map given x and y coordinates of 
    fixations and dimensions of the final ground truth map. 
    
    If `blur=False` a discrete ground truth map is computed. If `blur=True`, a 
    continuous map is created by smoothing over the discrete map with a Gaussian 
    kernel where the standard deviation is equal to `blur_factor`. `blur_factor` 
    should be equal to 1° of viewing angle.
    
    Args:
        x (1D array): x-coordinates (horizontal)
        y (1D array): y-coordinates (vertical)
        img_dims (tuple): dimensions for output map (vertical, horizontal),\
            corresponds to amount of pixels in each dimension. Defaults to (1080, 1920).
        blur (bool, optional): whether to return continous or discrete map.\
            Defaults to False (discrete fixation map).
        blur_factor (tuple, optional): standard deviation for Gaussian filter.\
            Dictates amount of smoothing. Defaults to [44.88, 38.41].
        fix_value (int, optional): Which value to fill in at points of fixations.\
            Defaults to 255.

    Returns:
        gt_map: ground truth saliency map
    """
    # convert coordinates to integer format for indexing
    x = np.array(x).round().astype(int)
    y = np.array(y).round().astype(int)

    # move out-of-bound coordinates inwards for indexing purposes
    x = np.where(x >= img_dims[1], img_dims[1] - 1, x)
    y = np.where(y >= img_dims[0], img_dims[0] - 1, y)

    # initialise array, then add non-zero values at fixation coordinates
    gt_map = np.zeros(img_dims, dtype=np.float32)
    for x_coord, y_coord in zip(x, y):
        # gt_map[y_coord, x_coord] = fix_value
        gt_map[y_coord, x_coord] += 1

    # blur over discrete array if continuous representation is desired
    if blur:
        gt_map = gaussian_filter(gt_map, sigma=blur_factor)

    return gt_map


# ------------------------------------------------------------------------------
# Metrics for evaluating the performance of saliency maps with fixation data and
# convenience function to do so for all saliency maps within a directory


def NSS(smap, fix_map):
    """Calculates the normalized scanpath saliency between a saliency map and a 
    map of fixation locations as the mean value of the normalized saliency map 
    at the fixation locations. Expects 2D arrays of equal dimensions.

    Args:
        smap (2D numpy.ndarray): Predicted (continuous) saliency map.
        fix_map (2D numpy.ndarray): Map of fixation locations (discrete). \
            All non-zero elements are treated as fixations.

    Returns:
        nss: normalized scanpath saliency
    """
    smap = (smap - smap.mean()) / smap.std()  # normalize by variance
    fix_map = (fix_map - fix_map.mean()) / fix_map.std()  # normalize by variance.

    # We could create a boolean map of all locations that were fixated.
    # Instead, we count the number of fixations per pixel and then z-score it.
    # This gives effectively the same map though.
    mask = fix_map > 0
    nss = smap[mask].mean()

    return nss


def evaluate_smaps_in_dir(dir, fix_map, metric="NSS", smap_format="png"):
    """Calculates performance scores (NSS or AUC) between all saliency maps that
    are saved within a directory and an array that contains the ground truth fixations 
    (created with `make_gt_smap`). Only saliency maps in the directory that end
    with the specified image format are taken into account.
    
    Args:
        dir (pathlib.Path or string): Path object or string that indicates \
            the location / directory where the saliency maps are stored.
        fix_map (2D numpy.ndarray): Map of fixation locations (discrete).
        metric (str, optional): Metric which should be used to evaluate performance. \
            Accepts "NSS" or "AUC". Defaults to "NSS".
        smap_format (str, optional): Image format in which the saliency maps are \
            saved as. Defaults to "png".
    
    Returns:
        results: Dictionary of the performance scores for all saliency maps within \
            the specified directory.
    """
    # convert dir to Path object if it isn't already
    if not isinstance(dir, pathlib.Path):
        dir = pathlib.Path(dir)

    # use Path object to iterate over all saliency maps in a directory and calculate
    # their performance scores
    results = {"Model": [], metric: []}
    for f in dir.glob(f"*.{smap_format}"):
        smap = load_smap(f)
        score = globals()[metric](smap, fix_map)
        results["Model"].append(f.stem)
        results[metric].append(score)

    return results


# ------------------------------------------------------------------------------
# Functions for to facilitate the creation of the baseline models

def simple_center_bias(img_dims=(1080, 1920), center_value=255):
    """Generates a simple center bias (center prior) with a two-dimensional
    Gaussian filter which is stretched to the shape of the array to include the 
    aspect ratio of the image. Useful for establishing a baseline model to compare
    the performance of saliency maps to.

    Args:
        img_dims (tuple): Dimensions for output center bias (vertical, horizontal),\
            corresponding to the number of pixels in each dimension. Defaults to (1080, 1920).
        center_value (int, optional): Which value to fill in at the approximate \
            center of the array. Defaults to 255.

    Returns:
        centerbias: numpy array for simple center bias
    """
    centerbias = np.zeros(img_dims, dtype=np.float32)  # initialize

    # assign approximate middle of array a non-zero value
    mid_x, mid_y = (round(i / 2) for i in img_dims)
    centerbias[mid_x, mid_y] = center_value

    # blur over array with 2 Gaussians stretched to the shape of the array
    centerbias = gaussian_filter(centerbias, sigma=(mid_x, mid_y))

    return centerbias


def oneobserver_perf(
    id, data, metric="NSS", id_col="ID", x_coord="x", y_coord="y",
):
    """Calculates performance score (NSS or AUC) to evaluate how well the fixations 
    from one observer predict the fixations from the n-1 other observers.
    The fixations from one observer are smoothed
    Coordinates of the fixations and the identifier should be all in one DataFrame.

    Args:
        id : The unique identifier for the one observer.
        data (DataFrame, optional): DataFrame containing fixation coordinates and \
            the unique identifier.
        metric (str, optional): Metric which should be used to evaluate performance. \
            Accepts "NSS" or "AUC". Defaults to "NSS".
        id_col (str, optional): Column name for the unique identifier. Defaults to "ID".
        x_coord (str, optional): Column containing x-coordinates of fixations. Defaults to "x".
        y_coord (str, optional): Column containing y-coordinates of fixations. Defaults to "y".
        blur_factor
    
    Returns:
        score: Performance score for the match between
    """
    one = data[data[id_col] == id]  # one observer
    rest = data[data[id_col] != id]  # n-1 other observers

    # create saliency and fixation maps to compare
    oneobserver_map = make_gt_smap(one[x_coord], one[y_coord], blur=True)
    rest_map = make_gt_smap(rest[x_coord], rest[y_coord])

    # calculate how well blurred over fixations from one observer predict the
    # fixations from n-1 other observers, using the metric of choice
    score = globals()[metric](oneobserver_map, rest_map)

    return score


def oneobserver_compare(
    id, data, limited_data, metric="NSS", id_col="ID", x_coord="x", y_coord="y",
):
    one = limited_data[limited_data[id_col] == id]  # one observer
    rest = data[data[id_col] != id]  # n-1 other observers

    # create saliency and fixation maps to compare
    oneobserver_map = make_gt_smap(one[x_coord], one[y_coord], blur=True)
    rest_map = make_gt_smap(rest[x_coord], rest[y_coord])

    # calculate how well blurred over fixations from one observer predict the
    # fixations from n-1 other observers, using the metric of choice
    score = globals()[metric](oneobserver_map, rest_map)

    return score


# ------------------------------------------------------------------------------
# Functions for easier and consistent visualisations
def vis_smap(
    arr,
    save_path="",
    cmap="inferno",
    dpi=600,
    overlay="",
    overlay_alpha=0.25,
    show_img=False,
    img_origin="lower",
):
    """Convenience function to visualise continuous saliency maps. Can also be
    used to generate and save high quality images from arrays in the same
    dimension as the input array.

    Args:
        arr (np.ndarray): input array to convert to image
        save_path (str, optional): Path to save image to. If length == 0, the image \
            won't be saved. Defaults to "".
        cmap (str, optional): Color map. Defaults to "inferno".
        dpi (int, optional): Resolution of the image. Defaults to 300.
        overlay (str|np.ndarray, optional): Path to image for background or\
            np.ndarray from the image. Defaults to "".
        overlay_alpha (float, optional): Tranparency factor for image to overlay.\
            Defaults to 0.25.
        show_img (bool, optional): Whether to display the figure. Defaults to True.
        img_origin (str, optional): Origin of the coordinate system in which to plot \
            the image. Can take "lower" or "upper". Defaults to "lower".
    """

    height, width = arr.shape
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set(xlim=[0, width], ylim=[height, 0])
    ax.imshow(arr, extent=[0, width, 0, height], origin=img_origin, cmap=cmap)

    # overlay background image if desired
    if isinstance(overlay, np.ndarray):
        overlay = Image.fromarray(overlay).transpose(Image.FLIP_TOP_BOTTOM)
        ax.imshow(
            overlay, cmap="gray", origin=img_origin, alpha=overlay_alpha,
        )
    elif isinstance(overlay, str) & len(overlay) != 0:
        overlay = load_smap(overlay)
        overlay = Image.fromarray(overlay).transpose(Image.FLIP_TOP_BOTTOM)
        ax.imshow(overlay, cmap="gray", origin=img_origin, alpha=overlay_alpha)

    # save figure if desired, given a nonempty image path
    if len(str(save_path)) != 0:
        fig.savefig(save_path, pad_inches=0, dpi=dpi)

    if show_img:
        plt.show()

    plt.close()


def vis_fixations(
    x,
    y,
    img_dims=(1080, 1920),
    save_path="",
    dpi=600,
    cbackground="black",
    cpoints="white",
    show_img=False,
):
    """Visualises the fixation points given two one-dimensional arrays that contain
    the x and y coordinates of the fixations. Fixations are plotted as a scatter plot
    in such a way that the size of the scatter dots is approximately equal to one
    pixel within the specified image dimensions.

    Args:
        x (1D array): x-coordinates (horizontal)
        y (1D array): y-coordinates (vertical)
        img_dims (tuple): Boundaries in which to plot the fixations. Should correspond \
            to the resolution of the image in pixels. Defaults to (1080, 1920).
        save_path (str, optional): Path to save image to. If length == 0, the image \
            won't be saved. Defaults to "".
        dpi (int, optional): Resolution of the image. Defaults to 300.
        cbackground (str, optional): Background color. Defaults to "black".
        cpoints (str, optional): Color of the fixations. Defaults to "white".
        show_img (bool, optional): Whether to display the figure. Defaults to True.
    """
    # produce plots in same dimension as another image
    figsize = img_dims[1] / float(dpi), img_dims[0] / float(dpi)

    fig = plt.figure(figsize=figsize, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set(xlim=[0, img_dims[1]], ylim=[0, img_dims[0]])
    ax.set_facecolor(cbackground)

    # set size argument to equal 1 pixel
    ax.scatter(x, y, color=cpoints, s=(72.0 / fig.dpi) ** 2, marker="o")

    # save figure if desired, given a nonempty image path
    if len(str(save_path)) != 0:
        fig.savefig(save_path, pad_inches=0, dpi=dpi)

    if show_img:
        plt.show()

    plt.close()


def vis_NSS_similarity(
    smap,
    fix_map,
    save_path="",
    cmap="inferno",
    blur_factor=(44.88, 44.88),
    overlay="",
    show_img=False,
    img_origin="lower",
):
    """Visualises where a saliency map does well or less well at capturing the
    actual ground truth saliency as given by ground truth fixation locations.
    
    The "goodness" of fit for a saliency map is computed by calculating 
    normalised scanpath saliency (NSS) values for each pixel in the input 
    saliency map. Pixels which do not overlap with fixation locations are 
    assigned a value of 0 (i.e. the mean value of the normalised saliency map). 
    As opposed to calculations in `NSS`, the mean is not taken to allow for 
    visualisation.
    
    For visualisation and creation of a heat map, the resulting array is blurred
    with a Gaussian filter where sigma is equal to 1° of visual angle 
    ((44.88, 38.41) in the NEMO data).

    Args:
        smap (2D array)): Predicted saliency map.
        fix_map (2D array)): Map of fixation locations. All non-zero elements are \
            treated as fixations.
        save_path (str, optional): Path to save image to. Gets passed to `vis_smap`.\
            Defaults to "". 
        cmap (str, optional): Color map. Defaults to "inferno".
        blur_factor (tuple, optional): standard deviation for Gaussian filter.\
            Dictates amount of smoothing. Defaults to (44.88, 38.41).
        overlay (str|np.ndarray, optional): Path to image for background or\
            np.ndarray from the image. Defaults to "". Gets passed to `vis_smap`.
        show_img (bool, optional): Whether to display the figure. Defaults to True.
        img_origin (str, optional): Origin of the coordinate system in which to plot \
            the image. Can take "lower" or "upper". Defaults to "lower".
    """
    smap = (smap - smap.mean()) / smap.std()  # normalize by variance

    # get NSS value at each fixation location, assign 0 to non-fixation locations
    mask = fix_map.astype(np.bool)
    nss_map = np.where(mask, smap, 0)

    # smooth over array and visualise
    nss_map = gaussian_filter(nss_map, sigma=blur_factor)
    vis_smap(
        nss_map,
        cmap=cmap,
        save_path=save_path,
        overlay=overlay,
        show_img=show_img,
        img_origin=img_origin,
    )


def vis_smaps_in_dir(
    dir,
    add_img=None,
    title_add_img="Free viewing image",
    ncols=3,
    save_path="",
    smap_format="png",
    output_dims=(8.27, 11.69),
    cmap="inferno",
    smap_names=None,
    show_img=False,
    smap_size=(1080, 1920),
):
    """Plots all images (saliency maps) in a directory and puts them in the same
    figure. Useful for conveniently showing all saliency maps in one go. If desired,
    another in-memory image (e.g. the free viewing image) can be added to upper 
    left corner of the figure.

    Args:
        dir (pathlib.Path or string): Path object or string that indicates \
            the location / directory where the saliency maps are stored.
        add_img (numpy.ndarray, optional): The in-memory image (numpy.ndarray) to \
            add to the figure in the upper left corner if desired. Defaults to None.
        title_add_img (str, optional): Title for the subfigure from add_img. \
            Defaults to "Free viewing image".
        ncols (int, optional): Number of columns in which to plot the saliency maps. \
            Defaults to 3.
        save_path (str, optional): Path to save image to. If length == 0, the image \
            won't be saved. Defaults to "".
        smap_format (str, optional): Image format in which the saliency maps are \
            saved as. Defaults to "png".
        output_dims (tuple, optional): Output dimensions of the figure. Defaults \
            to (8.27, 11.69), which corresponds to the dimensions of an A4 page in inches.
        cmap (str, optional): Color map. Defaults to "inferno".
        smap_names (array, optional): Array with custom titles for naming the subfigures 
            with the saliency maps. Has to have equal length to the number of saliency maps 
            in dir. If None, the stem of the image names are taken as the titles.
            Defaults to None. 
        show_img (bool, optional): Whether to display the figure. Defaults to True.
        smap_size (tuple, optional): Size of the saliency maps.
    """

    # load the baseline models
    if save_path.stem == "all_smaps":
        extra_path = Path.cwd() / "images" / "extras"
        image_paths = []
        image_paths.append(extra_path / "Fixations.png")
    else:
        image_paths = []

    # convert dir to Path object if it isn't already
    if not isinstance(dir, pathlib.Path):
        dir = pathlib.Path(dir)

    image_paths += sorted(list(dir.glob(f"*.{smap_format}")))

    # initialise figure and set output dimensions
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(output_dims[0], output_dims[1], forward=True)
    ax = []

    # dynamically determine number of rows to fit images within plot, given the
    # number of images to fit
    nrows = np.ceil(len(image_paths) / ncols).astype(int)

    # add additional image in front of all other images, if desired
    loc = 1
    if add_img is not None:
        nrows = np.ceil((len(image_paths) + 1) / ncols).astype(int)
        ax.append(fig.add_subplot(nrows, ncols, loc))
        ax[-1].set_title(title_add_img)
        ax[-1].set_axis_off()
        plt.imshow(add_img)
        loc += 1

    # add all saliency maps in directory to the figure, naming them by the stem
    # of the file name
    for i, f in enumerate(image_paths):
        smap = load_smap(f)
        ax.append(fig.add_subplot(nrows, ncols, i + loc))
        if smap_names is not None:
            ax[-1].set_title(smap_names[i])
        else:
            ax[-1].set_title(f.stem)
        ax[-1].set_axis_off()
        plt.imshow(smap, cmap=cmap, extent=[0, smap_size[1], 0, smap_size[0]], origin='lower')

    plt.subplots_adjust(wspace=0, hspace=0)

    if show_img:
        plt.show()

    if len(str(save_path)) != 0:
        fig.savefig(save_path, dpi=600, facecolor="white")

    plt.close()
