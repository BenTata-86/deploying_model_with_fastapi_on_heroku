import pandas as pd
import numpy as np
import os



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
    df = df.drop(columns=["capital-gain", "capital-loss"])

    #Replace na with 'unknown'
    df = df.replace(np.nan, "unknown")
    return df



def main():

    path = "s3://tatacensus/14/5de00f6e6053d3f7044628f9a5b5ff"
    df= basic_cleaning(path)
    print("importing data...")
    ROOT_DIR = os.path.dirname(os.path.abspath("/home/bshegitim1/udacity_mlops/deploying_model_with_fastapi_on_heroku/src"))
    filename = f'{ROOT_DIR}/data/cooked_data.csv'
    df.to_csv(filename)
    print("save ..")

        

if __name__ == "__main__":
    main()