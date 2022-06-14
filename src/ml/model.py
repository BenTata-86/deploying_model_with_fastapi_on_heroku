from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from . import preprocess_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : Random Forest Classifier
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50,100,200],
        'max_features': [5,6],
        'max_depth': [6,8]
        }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)


    model = cv_rfc.best_estimator_
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Random Forest Classifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def compute_score_per_slice(model, data, encoder,
                            lb, cat_features):
    """
    Compute score in a category class slice
    Parameters:
    
    model
    data
    encoder
    lb

    Returns:
    metrics_dict
    """
    metrics_dict = {}
    for cls in data['marital-status']:
        temp_df = data[data['marital-status'] == cls]

        slice_feature, slice_label, _, _ = preprocess_data.process_data(
                    temp_df,
                    categorical_features=cat_features, training=False,
                    label="salary", encoder=encoder, lb=lb)

        slice_pred = model.predict(slice_feature)

        prc, rcl, fb = compute_model_metrics(slice_label, slice_pred)

        metrics_dict[cls] = {"precision": prc, "recall": rcl, "fbeta": fb}
    return metrics_dict
        


