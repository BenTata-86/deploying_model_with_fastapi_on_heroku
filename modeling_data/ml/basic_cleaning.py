import pandas as pd
import numpy as np

def basic_cleaning(path):
    '''
    Takes path and read csv file and apply basic cleaning on data

    args:
        path : path for input data
    '''

    #Read data from path
    df = pd.read_csv(path, skipinitialspace=True ,na_values='?')

    #Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    #Drop capital_gain, capital_loss
    df = df.drop(columns=["Capital_gain", "Capital_loss"])

    #Replace na with 'unknown'
    df = df.replace(np.nan, "unknown")

    filename = 'data/cooked_data.csv'
    df.to_csv(filename)