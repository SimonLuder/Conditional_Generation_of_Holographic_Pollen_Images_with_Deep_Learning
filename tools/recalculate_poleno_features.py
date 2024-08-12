'''
This script is used to recalculate the visual features from the pollen objects in the holographic images. 

Two arguments are required for the recalculation:

    - database: path of the poleno_marvel.db file
    - image_folder: path to the folder containing the holographic images

The default path configuration is setup to run inside the singularity container. To run it localy the two precious paths need to be adapted.
'''

import os
import sys
import sqlite3
import argparse
import pandas as pd

# add parent dir to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) 
sys.path.append(parent_dir)

from utils.data_processing import recalculate_holographic_features

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for resnet training')
    parser.add_argument('--database', default='/workspace/data/Poleno/poleno_marvel.db', type=str)
    parser.add_argument('--image_folder', default='/workspace/data/Poleno', type=str)
    args = parser.parse_args()

    database = args.database
    image_folder = args.image_folder

    print(database, image_folder)

    # Connect to the SQLite database
    conn = sqlite3.connect(database)

    # get table computed_data_full as dataframe
    computed_data_full = pd.read_sql_query("SELECT * FROM computed_data_full", conn)

    # close the database connection
    conn.close()

    # recalculate features for all images in the dataframe
    computed_data_full_recalc = recalculate_holographic_features(computed_data_full, image_folder)

    # save the dataframe
    computed_data_full_recalc.to_csv("computed_data_full_re.csv", index=False)