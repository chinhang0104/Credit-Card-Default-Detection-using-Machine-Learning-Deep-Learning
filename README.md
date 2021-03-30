# Credit-Card-Default-Detection-using-Machine-Learning-Deep-Learning

## Dependencies
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
```
pip install numpy, pandas, seaborn, matplotlib, tensorflow, sklearn, scipy
```
## Abstract
Using [data from Kaggle](https://www.kaggle.com/hassanamin/uci-credit-card), we are using various machine learning and deep learning models to predict credit card default. The final output ensembles Random Forest Classifier, Deep Neural Networks, Logistic Regression, Support Vector Machine and XGB Classifier.

## Sections
### Data Preparation
We first employ min_max_scaler to normalize data. Then, use SMOTE to handle imbalanced class. Finally, training data is split into 80% training and 20% validation. 

### Modeling
We employed grid search cross validation to search best hyperparameters. Auc score is used for matrix evaluation.

### Model Ensemble
We ensemble the above models with finetuned hyperparameter. Final output is based on weighted average of the models. 
