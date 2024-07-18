import os
import cv2
import skimage
import numpy as np
from tqdm import tqdm

def recalculate_holographic_features(df, image_path):
    """
    Recalculates holographic features for each row in the DataFrame.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame containing relevant data.
    - image_path (str): Path to the directory containing images.

    Returns:
    - pandas.DataFrame: Updated DataFrame with recalculated features.
    """

    df = df.copy()

    for i, row in tqdm(df.iterrows(), total=len(df)):

        img_path = os.path.join(image_path, row["dataset_id"], row["rec_path"])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            rp = regionprops_from_numpy(img)
            df.at[i, 'area']                  = rp[0].area
            df.at[i, 'bbox_area']             = rp[0].bbox_area
            df.at[i, 'convex_area']           = rp[0].convex_area
            df.at[i, 'major_axis_length']     = rp[0].major_axis_length
            df.at[i, 'minor_axis_length']     = rp[0].minor_axis_length
            df.at[i, 'eccentricity']          = rp[0].eccentricity
            df.at[i, 'solidity']              = rp[0].solidity
            df.at[i, 'perimeter']             = rp[0].perimeter
            df.at[i, 'perimeter_crofton']     = rp[0].perimeter_crofton
            df.at[i, 'equivalent_diameter']   = rp[0].equivalent_diameter
            df.at[i, 'orientation']           = rp[0].orientation
            df.at[i, 'feret_diameter_max']    = rp[0].feret_diameter_max
            df.at[i, 'max_intensity']         = rp[0].intensity_max[0]
            df.at[i, 'min_intensity']         = rp[0].intensity_min[0]
            df.at[i, 'mean_intensity']        = rp[0].intensity_mean[0]

        else:
            df.at[i, 'area']                  = None
            df.at[i, 'bbox_area']             = None
            df.at[i, 'convex_area']           = None
            df.at[i, 'major_axis_length']     = None
            df.at[i, 'minor_axis_length']     = None
            df.at[i, 'eccentricity']          = None
            df.at[i, 'solidity']              = None
            df.at[i, 'perimeter']             = None
            df.at[i, 'perimeter_crofton']     = None
            df.at[i, 'equivalent_diameter']   = None
            df.at[i, 'orientation']           = None
            df.at[i, 'feret_diameter_max']    = None
            df.at[i, 'max_intensity']         = None
            df.at[i, 'min_intensity']         = None
            df.at[i, 'mean_intensity']        = None

    return df



def convert_pixel_to_um(df, resolution):
    """
    Convert the measurements of an object in an image from pixels to micrometers (um).

    This function takes a DataFrame containing measurements in pixels and a resolution
    in pixels per um, and returns a new DataFrame where all measurements are converted
    to um.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the measurements in pixels. 
      It should have columns for 'area', 'bbox_area', 'convex_area', 'major_axis_length', 
      'minor_axis_length', 'perimeter', 'perimeter_crofton', 'equivalent_diameter', and 
      'feret_diameter_max'.
    - resolution (float): The resolution of the image in pixel per um.

    Returns:
    - pandas.DataFrame: A new DataFrame with the same structure as the input, but where 
      all measurements are converted to um.
    """
    df_um = df.copy()
    df_um['event_id']               = df['event_id']
    df_um['dataset_id']             = df['dataset_id']
    df_um['label']                  = df['label']
    df_um['rec_path']               = df['rec_path']
    df_um['image_nr']               = df['image_nr']
    df_um['area']                   = get_area_in_um2(df['area'], resolution)
    df_um['bbox_area']              = get_bbox_area_in_um2(df['bbox_area'], resolution)
    df_um['convex_area']            = get_convex_hull_area_in_um2(df['convex_area'], resolution)
    df_um['major_axis_length']      = get_major_axis_in_um(df['major_axis_length'], resolution)
    df_um['minor_axis_length']      = get_minor_axis_in_um(df['minor_axis_length'], resolution)
    df_um['eccentricity']           = df['eccentricity']
    df_um['solidity']               = df['solidity']
    df_um['perimeter']              = get_perimeter_in_um(df['perimeter'], resolution)
    df_um['perimeter_crofton']      = get_crofton_perimeter_in_um(df['perimeter_crofton'], resolution)
    df_um['equivalent_diameter']    = get_equivalent_diameter_in_um(df['equivalent_diameter'], resolution)
    df_um['orientation']            = df['orientation']
    df_um['feret_diameter_max']     = get_feret_diameter_in_um(df['feret_diameter_max'], resolution)
    df_um['max_intensity']          = df['max_intensity']
    df_um['min_intensity']          = df['min_intensity']
    df_um['mean_intensity']         = df['mean_intensity']
    return df_um


def regionprops_from_numpy(img):

    img_intensity = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    thresh = get_grain_mask_from_holo(img)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [max(contours, key=cv2.contourArea)] # assumption: pollen consist of one connected piece

    img_masked = np.zeros_like(img, dtype=np.uint8)
    img_masked = cv2.drawContours(img_masked, contours, -1, (255, 255, 255), -1)

    rp = skimage.measure.regionprops(label_image=img_masked, intensity_image=img_intensity, cache=True)

    return rp

# ↓↓↓ The Code below was nicely provided by Roman Studer ↓↓↓

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


def get_holo_resolution():
    """
    Calculate the resolution of the holographic image in pixel per µm.
    The resolution is fixed across all devices and it can be assumed, that one pixel has h and w 0.595 µm.
    """
    return 1 / 0.595


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
