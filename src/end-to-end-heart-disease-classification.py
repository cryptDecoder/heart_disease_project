# Predicting heart disease using machine learning

'''
This project looks into using various Python-based machine learning and data science libraries in an attempt to build
 a machine learning model capable of predicting whether someone has heart disease based on their medical
 attributes.
'''
from matplotlib.transforms import Bbox

'''
We are going to take following approach
1: Problem definition
2: Data
3: Evaluation
4: Features
5: Modelling
6: Experimentation
'''

## 1: Problem definition

'''
Given clinical parameters about a patient, can we predict whether or not have heart disease
'''
## 2: Data

'''
The original data came from the Cleavland data from the UCI machine learning Repository
There is also a version of it available on kaggle
'''

## 3: Evaluation

## 4: Features
'''
This is where you'll get different information about each of the features in your data.
Create data dictionary
'''

######## ********** *****************  ************ ************* ##############

# Preparing the Tools
print("######### Preparing the tools ########")
# import all the tools we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import model form Scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

print("all import done .........\n")
## Load the data
print("loading the data")
df = pd.read_csv('../data/heart.csv')
print(df.head())
print(df.shape)

## data exploration
'''
The goal here is to find out more about the data and become
a subject export on the dataset you're working with 
1 : What questions are you trying to solve?
2: what kind of the data we have and how do we deal with it?
3: where's missing from the data and how do we deal with it?
4: where are the outliers and why should you care about theme
5: how can add, change or remove features to get more out of your data
'''
print("data exploration....\n")
print(df.tail())

# let's find out how many of each class there

print(df['target'].value_counts())
print("plotting the values :")
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue'])
# plt.show()

# data info
print(df.info())
# check missing values
print(df.isna().sum())

# describe
print(df.describe())

# heart disease frequency according to gender
print(df.sex.value_counts())

# compare target column to sex column
print(pd.crosstab(df.target, df.sex))

pd.crosstab(df.target, df.sex).plot(kind='bar', figsize=(10, 6), color=['salmon', 'lightblue'])
plt.title(" Heart disease frequency for gender")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(['female', 'male'])
plt.xticks(rotation=0)
# plt.show()

# Age vs max heart for heart disease

# creating another figure
plt.figure(figsize=(10, 6))

# scatter with positive examples
plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c='salmon')
# plt.show()

# scatter with negative example
plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0], c='lightblue')

# Add helpful info
plt.title("Heart Disease in function of age and Max heart rate")
plt.xlabel("Age")
plt.ylabel("Max heart rate")
plt.legend(['disease', 'No Disease'])
# plt.show()

# check the distribution of the age column with a histogram
# df.age.plot.hist()
# plt.show()


# heart disease frequency per chest pain type
'''
cp: chest pain type
-- Value 1: typical angina
-- Value 2: atypical angina
-- Value 3: non-anginal pain
-- Value 4: asymptomatic
'''

print("\n")
print(pd.crosstab(df.cp, df.target))

# make the crosstab more visual

pd.crosstab(df.cp, df.target).plot(kind='bar', figsize=(10, 6), color=['salmon', 'lightblue'])

# add some communication
plt.title("Heart Disease frequency per chest pain")
plt.xlabel("Chest pain type")
plt.ylabel("Amount")
plt.legend(['disease', 'No Disease'])
plt.xticks(rotation=0)
# plt.show()


# Building correlation metrics
print(df.corr())

# let's make our correlation matrix a little prettier

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
# plt.show()

## Modelling
print(df.head())

X = df.drop('target', axis=1)
y = df['target']
print("print X:\n", X)
print("print y\n", y)

np.random.seed(42)
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.head())
print(y_train.head())

# now our data is splited into a train and test model now it's time to build machine learning model
# choosing the right  machine learning model
'''

    We are going to test three machine learning model
    1: Logistic Regression
    2: K-Nearest Neighbours Classifier
    3: Random Forest Classifier
    
'''
# put models in dictionary
models = {"Logistics Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}


# create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluate the given machine learning models
    :param models: a dict of different Scikit-Learn machine learning model
    :param X_train: Training data (no labels)
    :param X_test: Testing data with no labels
    :param y_train: training labels
    :param y_test: testing labels
    :return: model fit and model score
    '''
    # set random seed
    np.random.seed(42)

    # make a dictionary to keep model scores
    model_score = {}
    # loop through the model
    for name, model in models.items():
        # fit the model ti the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_score[name] = model.score(X_test, y_test)
    return model_score

    pass


models_scores = fit_and_score(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print(models_scores)

# model comparison
model_compare = pd.DataFrame(models_scores, index=['accuracy'])
print(model_compare)
model_compare.plot.bar();
# plt.show()

# Tuning/improving our model
'''
    Now we have got baseline model...
    and we know a model's first predications aren't always what we should based our next steps off.
    What should do?
    Let's look at the following
    1: Hyper parameter tuning
    2: Features importance 
    3: confusion matrix
    4: Cross-validation
    5: Precision
    6: Recall
    7: Classification report
    8: ROC curve
    9: Area under the curve (AUC)
'''

# Hyper parameter Tuning
train_score = []
test_score = []

# create a list of different values of n neighbours
neighbors = range(1, 21)
# set up KNN
knn = KNeighborsClassifier()

for i in neighbors:
    knn.set_params(n_neighbors=i)
    # fit the algorithm
    knn.fit(X_train, y_train)
    # update the training score list
    train_score.append(knn.score(X_train, y_train))
    # update the test score
    test_score.append(knn.score(X_test, y_test))
print(train_score)
print(test_score)

plt.plot(neighbors, train_score, label="Train Score")
plt.plot(neighbors, test_score, label="Test Score")
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
print(f"Maximum KNN score on the test data :{max(test_score) * 100:.2f}%")
# plt.show()

# hyper parameter tuning with RandomizedSearchCV
'''
 we are going to tune:
    1: LogisticsRegression
    2: RandomForestClassifier
    
    ... using randomizedSearchCV
'''
# create a hyper parameter grid for logisticsRegression

log_reg_grid = {
    'C': np.logspace(-4, 4, 20),
    'solver': ['liblinear']
}
print(log_reg_grid)
# create hyper parameter grid for RandomForestClassifier
rf_grid = {
    "n_estimators": np.arange(10, 1000, 50),
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": np.arange(2, 20, 2),
    "min_samples_leaf": np.arange(1, 20, 2)
}

print(rf_grid)

'''
 now we have got hyper parameter grids setup for each of our models,
 let's tune them using RandomizedSearchCV
'''

# Tune logistics regression

# set up random seed
np.random.seed(42)

# set up rand hyperparameter  search for logisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True
                                )

# fit random hyper parameter search model for logisticRegression

rs_log_reg.fit(X_train, y_train)
print(rs_log_reg.best_params_)
# print the score of rs_log_reg
print(rs_log_reg.score(X_test, y_test))

'''
    Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()
'''

# set up random seed

np.random.seed(42)
# Setup random hyperparameter search for RandomForestClassifier

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)
# Fit random hyper parameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)

# Find the best hyper parameter
print(rs_rf.best_params_)

# Evaluate the randomized search RandomForestClassifier model

print(rs_rf.score(X_test, y_test))

# hyper parameter tuning with GridSearchCV

'''
Since our LogisticRegression model provides the best score so far,
we'll try and improve them again using GridSearchCV...
'''

# different hyper parameter for our LogisticRegression Model
log_reg_grid = {
    "C": np.logspace(-4, 4, 30),
    'solver': ['liblinear']
}

# setup grid hyperparameter search for LogisticRegression

gs_log_reg = GridSearchCV(
    LogisticRegression(),
    param_grid=log_reg_grid,
    cv=5,
    verbose=True
)

# fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train)

# check the best hyper parameter

print(gs_log_reg.best_params_)

# evaluate the grid search LogisticRegression model
print(gs_log_reg.score(X_test, y_test))

# Evaluating our tuned machine learning classifier,beyond accuracy
'''
 * ROC curve and AUC score
 * Confusion matrix
 * Classification report
 * Precision 
 * Recall
 * F1-Score
  ... and it would be great if cross-validation was used where possible.
  To make comparisons and evaluate our trained model, first we need to make predictions
'''

# Make predictions with tuned model
y_pred = gs_log_reg.predict(X_test)
print(y_pred)

# plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test)
# plt.show()

# confusion metrix

print(confusion_matrix(y_test, y_pred))
sns.set(font_scale=1.5)


def plot_conf_mat(y_test, y_pred):
    '''
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    :param y_test: true label
    :param y_pred: predicted label
    :return: Confusion matrix returns the following terms
            * true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
            * true negatives (TN): We predicted no, and they don't have the disease.
            * false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
            * false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
    '''
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False)
    plt.xlabel("True label")
    plt.ylabel("predicted label")
    plt.show()
    pass


# plot_conf_mat(y_test, y_pred)

'''
 Now we've got a ROC curve, an metric and a confusion matrix,
 let's get a classification report as well as cross-validated
 precision, recall and f1-score.
'''

# check classification report
print(classification_report(y_test, y_pred))

'''
 Calculate evaluation metrics using cross-validation 
 we're going to calculate accuracy,precision, recall and f1-score of our model using cross-validation and to
 do so we'll be using cross-val-score().
'''

# check best hyper parameter
print(gs_log_reg.best_params_)

# create a new classifier with best parameter
clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')

# cross-validated accuracy
cv_acc = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='accuracy'
)
print(cv_acc)
print(np.mean(cv_acc))
cv_acc = np.mean(cv_acc)

# Cross-validated precision

cv_precision = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='precision'
)
print(cv_precision)
print(np.mean(cv_precision))
cv_precision = np.mean(cv_precision)

# Cross-validated  recall

cv_recall = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='recall'
)
print(cv_recall)
print(np.mean(cv_recall))
cv_recall = np.mean(cv_recall)

# cross-validated f1-score

cv_f1 = cross_val_score(
    clf,
    X,
    y,
    cv=5,
    scoring='f1'
)
print(cv_f1)
print(np.mean(cv_f1))
cv_f1 = np.mean(cv_f1)

# visualize cross-validated metrics
cv_metrics = pd.DataFrame({
    "accuracy": cv_acc,
    "Precision": cv_precision,
    "Recall": cv_recall,
    "F1": cv_f1
}, index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics ", legend=False)
# plt.show()

'''
    Features importance
    * Feature importance is another as asking, "Which features contributed most
    to the outcomes of the model and how they contribute.
    *  Finding feature importance is different for each machine learning model. 
    one way to find feature importance is to search for "(MODEL NAME) feature importance"  
    
    ... let's find the feature importance for our LogisticRegression model. 
'''

# fit and instance of LogisticRegression
print(df.head())
clf = LogisticRegression(C=0.20433597178569418, solver='liblinear')
clf.fit(X_train, y_train)

# check the coef_
print(clf.coef_)

# match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
print(feature_dict)

# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False)
# plt.show()

'''
 ** Experimenting **
    if you haven't hit your evaluation metric yet ... ask yourself ...
    * Could you collect more data?
    * Could you try better model? Like CatBoost or XGBoost?
    * Could you improve the current models? (beyond what we've done so far)
    * if your model is good enough ( you have hit your evaluation metric)
    how would you export it and share it with others?
    
'''
