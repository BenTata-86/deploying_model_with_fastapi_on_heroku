import os
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split

from ml.preprocess_data import process_data
from ml.model import *




def main():

    # Define categorical Features.
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    #Setting root directory
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    #Setting data & model directory
    data_pth = f'{ROOT_DIR}/data'
    model_pth = f'{ROOT_DIR}/model'
    #Load data
    train_data = pd.read_csv(f"{data_pth}/train.csv")
    test_data = pd.read_csv(f"{data_pth}/test.csv")

    #Load Model & encoders
    model = load(f"{model_pth}/model.pkl")
    encoder = load(f"{model_pth}/encoder.pkl")
    lb = load(f"{model_pth}/lb.pkl")
    #Process Data
    X_train, y_train, _,_ = process_data(
        train_data, cat_features,training=False ,encoder=encoder, lb=lb)
    X_test, y_test, _,_ = process_data(
        test_data, cat_features, training=False, encoder=encoder, lb=lb)
    
    #Get inferences on datas
    train_preds = inference(model=model, X=X_train)
    test_preds = inference(model=model, X=X_test)
    #Get model metrics on datas

    train_prc, train_rcl, train_beta = compute_model_metrics(
        y=y_train, preds=train_preds
    )
    training_metric = "Precision: %s " \
                              "Recall: %s FBeta: %s" %(train_prc, train_rcl, train_beta)
    test_prc, test_rcl, test_beta = compute_model_metrics(
        y=y_test, preds=test_preds
    )
    testing_metric = "Precision: %s " \
                              "Recall: %s FBeta: %s" % (test_prc, test_rcl, test_beta)
    with open(f'{model_pth}/output.txt', 'w') as f:
        f.write(training_metric + '\n')
        f.write(testing_metric + '\n')


    

if __name__ == "__main__":
    main()