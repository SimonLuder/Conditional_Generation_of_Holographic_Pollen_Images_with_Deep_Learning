import os
import sys
import cv2
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from utils.data_processing import recalculate_holographic_features

if __name__ == "__main__":
    
    # database = 'Z:\marvel\marvel-fhnw\data\Poleno/poleno_marvel.db'
    # image_folder = 'Z:\marvel\marvel-fhnw\data\Poleno'

    database = '/workspace/data/Poleno/poleno_marvel.db'
    image_folder = '/workspace/data/Poleno'

    # Connect to the SQLite database
    conn = sqlite3.connect(database)

    # get table computed_data_full as dataframe
    computed_data_full = pd.read_sql_query("SELECT * FROM computed_data_full", conn)

    # close the database connection
    conn.close()

    # recalculate features for all images in the dataframe
    computed_data_full_recalc = recalculate_holographic_features(computed_data_full, image_folder)

    # # convert metrics with units in pixel to μm
    # resolution = get_holo_resolution()
    # computed_data_full_recalc = data_processing.convert_pixel_to_um(computed_data_full_recalc, resolution)

    # save the dataframe
    computed_data_full_recalc.to_csv("computed_data_full_re.csv", index=False)