# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Develop by Taha A. for Udacity MLops Nanodegree Type: Random Forest Classifier

## Intended Use
This model is used to predict Income whether it is >50K USD or <50K USD based on
census data : https://archive.ics.uci.edu/ml/datasets/census+income
## Training Data
.80 percent of data is used for training and validation with 5 fold

## Evaluation Data
.20 percent of data is used for evaluation
'marital-status' classes is used for evaluating the slices of data
## Metrics
The performance of the model is measured with the metrics : Precision, Recall
and F1 Beta score.
## Ethical Considerations
The model was also evaluated in 'marital-status' as well for ethical considerations
## Caveats and Recommendations
The data have some class imbalances and this may indicate biases on those classes