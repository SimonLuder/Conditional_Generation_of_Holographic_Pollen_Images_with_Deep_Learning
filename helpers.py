import glob
import os
import sqlite3

import cv2
import numpy as np
import pandas as pd


def get_db_connection(path="./data/Poleno/poleno_marvel.db"):
    return sqlite3.connect(path)


def get_em_images(
    path="Z:\marvel\marvel-fhnw\data\Paldat\images_subset_poleno_classes",
):
    """
    Get all images from the target folder (including subfolders)
    Function expects the following folder structure:
     - images/
      - genus/ (g_*)
        - species/ (s_*)
          - images (*.jpg)
    Returns a dictionary with the following structure:
    {
    "genus": {
        "species": {
            "image_name": "path_to_image"
        }
    }

    :param path: Path to the folder containing the images
    :return: Dictionary with the structure described above
    """
    images = {}
    for genus in glob.glob(os.path.join(path, "g_*")):
        genus_name = os.path.basename(genus)
        images[genus_name] = {}
        for species in glob.glob(os.path.join(genus, "s_*")):
            species_name = os.path.basename(species)
            images[genus_name][species_name] = {}
            for image in glob.glob(os.path.join(species, "*.jpg")):
                image_name = os.path.basename(image)
                images[genus_name][species_name][image_name] = image
    return images


def get_grain_mask_from_em(img: np.ndarray):
    """
    Takes a raw electron microscope image and returns a binary mask of the grain area. Useful for segmentation or background removal

    :param img: Electron microscope image
    :return: Binary mask of the grain area
    """
    # apply threshold
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [max(contours, key=cv2.contourArea)]

    img = np.zeros_like(img)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    cv2.fillPoly(img, contours, color=(255, 255, 255))

    return img


def get_em_image_size(img):
    """
    Calculate the area of the image in µm^2 based on the identification of a white horizontal scale bar.

    Parameters:
    - img (array-like): The image in pixel format.

    Returns:
    - float: The area of the image in µm^2 if a scale bar is found, else None.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate white regions
    _, thresh = cv2.threshold(gray, 249, 255, cv2.THRESH_BINARY)

    # Use morphological operations to ensure horizontal line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  #
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the scale bar is the largest contour after morphological operations
    scale_bar = max(contours, key=cv2.contourArea)

    _, _, w_pixels, _ = cv2.boundingRect(scale_bar)
    scale_bar_length_pixels = w_pixels
    resolution = scale_bar_length_pixels / 10  # scale bar represents 10 µm

    height, width = img.shape[:2]

    # Convert dimensions to µm
    width_um = width / resolution
    height_um = height / resolution

    # Calculate area in µm^2
    area_um2 = width_um * height_um

    return area_um2


def get_relative_grain_area_from_em(img):
    area = get_area_in_px_from_em(img)
    return area / (img.shape[0] * img.shape[1])


def get_area(img):
    """Returns the number of white pixels in the image. Assumes the image is binarized.

    :param img: np.array
    :return: int
    """
    return np.sum(img == 255)


def get_relative_area(img):
    """
    Returns the relative area of the image. Assumes the image is binarized and has only one contour.
    """
    area = get_area(img)
    return area / (img.shape[0] * img.shape[1])


def get_grain_area_in_um(relative_area, em_image_size):
    """
    Calculate the grain area in µm^2 based on the relative area and the image size.

    Parameters:
    - relative_area (float): The relative area of the grain.
    - em_image_size (float): The size of the EM image in µm^2.

    Returns:
    - float: The grain area in µm^2.
    """
    return relative_area * em_image_size


def get_area_in_px_from_em(img: np.ndarray):
    """
    Takes a raw electron microscope image and returns the area of the grain in pixels. Useful for grain size estimation.

    :param img: Electron microscope image
    :return: Area of the grain in pixels
    """
    mask = get_grain_mask_from_em(img)
    return np.sum(mask == 255)


def load_image_from_sql(
    row: pd.Series, base: str = "Z:\marvel\marvel-fhnw\data\Poleno"
):
    """Loads an image from a row in the database.
    Row must contain the columns dataset_id and rec_path.

    :param base: Base path to the images
    :param row: Row from the database
    :return: Image
    """
    dataset_id, rec_path = row["dataset_id"], row["rec_path"]
    path = str(os.path.join(base, f"{dataset_id}", rec_path))
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_grain_mask_from_holo(img: np.ndarray):
    """
    Takes a raw hologram image and returns a binary mask of the grain area. Useful for segmentation or background removal
    Use with care ;)

    :param img: Hologram image
    :return: Binary mask of the grain area
    """
    # apply threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert mask
    mask = cv2.bitwise_not(mask)
    return mask


def get_grain_area_in_px_from_holo(img: np.ndarray):
    """
    Takes a raw hologram image and returns the area of the grain in pixels. Useful for grain size estimation.
    Use with care ;)

    :param img: Hologram image
    :return: Area of the grain in pixels
    """
    mask = get_grain_mask_from_holo(img)
    return np.sum(mask == 0)


def grain_area_px_to_um_from_holo(area_px: float, px_size: float = 0.595):
    """
    Converts the area of a grain from pixels to micrometers.

    :param area_px: Area of the grain in pixels
    :param px_size: Size of a pixel in micrometers
    :return: Area of the grain in micrometers
    """
    return area_px * px_size**2


def get_holo_resolution():
    """
    Calculate the resolution of the holographic image in pixel per µm.
    The resolution is fixed across all devices and it can be assumed, that one pixel has h and w 0.595 µm.
    """
    return 1 / 0.595


def get_em_resolution(img):
    """
    Calculate the resolution of the EM image in pixel per µm.
    NOTE: The resolution can vary between images use this function for each image individually.

    Parameters:
    - img (array-like): A raw em image in pixel format.

    Returns:
    - float: The resolution of the image in pixel per µm.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate white regions
    _, thresh = cv2.threshold(gray, 249, 255, cv2.THRESH_BINARY)

    # Use morphological operations to ensure horizontal line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  #
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the scale bar is the largest contour after morphological operations
    scale_bar = max(contours, key=cv2.contourArea)

    _, _, w_pixels, _ = cv2.boundingRect(scale_bar)
    scale_bar_length_pixels = w_pixels
    resolution = scale_bar_length_pixels / 10  # scale bar represents 10 µm

    return resolution


def get_major_axis_in_um(major_axis_length, resolution):
    """
    Calculate the major axis length in µm based on the pixel length and the resolution.

    Parameters:
    - major_axis_length (float): The major axis length in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The major axis length in µm.
    """
    return major_axis_length / resolution


def get_minor_axis_in_um(minor_axis_length, resolution):
    """
    Calculate the minor axis length in µm based on the pixel length and the resolution.

    Parameters:
    - minor_axis_length (float): The minor axis length in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The minor axis length in µm.
    """
    return minor_axis_length / resolution


def get_bbox_area_in_um2(bbox_area, resolution):
    """
    Calculate the area of the bounding box in µm^2 based on the pixel area and the resolution.

    Parameters:
    - bbox_area (float): The area of the bounding box in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The area of the bounding box in µm^2.
    """
    return bbox_area / resolution**2


def get_area_in_um2(area, resolution):
    """
    Calculate the area in µm^2 based on the pixel area and the resolution.

    Parameters:
    - area (float): The area in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The area in µm^2.
    """
    return area / resolution**2


def get_convex_hull_area_in_um2(convex_hull_area, resolution):
    """
    Calculate the area of the convex hull in µm^2 based on the pixel area and the resolution.

    Parameters:
    - convex_hull_area (float): The area of the convex hull in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The area of the convex hull in µm^2.
    """
    return convex_hull_area / resolution**2


def get_perimeter_in_um(perimeter, resolution):
    """
    Calculate the perimeter in µm based on the pixel perimeter and the resolution.

    Parameters:
    - perimeter (float): The perimeter in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The perimeter in µm.
    """
    return perimeter / resolution


def get_crofton_perimeter_in_um(crofton_perimeter, resolution):
    """
    Calculate the Crofton perimeter in µm based on the pixel perimeter and the resolution.

    Parameters:
    - crofton_perimeter (float): The Crofton perimeter in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The Crofton perimeter in µm.
    """
    return crofton_perimeter / resolution


def get_equivalent_diameter_in_um(equivalent_diameter, resolution):
    """
    Calculate the equivalent diameter in µm based on the pixel diameter and the resolution.

    Parameters:
    - equivalent_diameter (float): The equivalent diameter in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The equivalent diameter in µm.
    """
    return equivalent_diameter / resolution


def get_feret_diameter_in_um(feret_diameter, resolution):
    """
    Calculate the Feret diameter in µm based on the pixel diameter and the resolution.

    Parameters:
    - feret_diameter (float): The Feret diameter in pixels.
    - resolution (float): The resolution of the image in pixel per µm.

    Returns:
    - float: The Feret diameter in µm.
    """
    return feret_diameter / resolution
