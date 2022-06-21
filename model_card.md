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
precision:0.7757909215955984
recall:0.3863013698630137
fbeta:0.5157750342935529
## Ethical Considerations
The model was also evaluated for all different classes. Since there are classes
with less data, the results for the less represented classes are not predicted 
well. For example "Married-civ-spouse" class in "marital-status" is a well represented
class and the prediction results are close to the overall metrics. However "Divorced"
has a precision: 1.0 but recall and f1 beta scores are : 0 due to its class is
a less represented class. This should be kept in mind for Ethical Considerations
## Caveats and Recommendations
The data have some class imbalances and this may indicate biases on those classes