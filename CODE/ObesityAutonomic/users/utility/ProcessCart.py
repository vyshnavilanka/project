import pandas as pd
from sklearn.model_selection import train_test_split
from django.conf import settings
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation

from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples
from sklearn.metrics import classification_report  # for model evaluation metrics
from sklearn import tree  # for decision tree models

import plotly.express as px  # for data visualization
import plotly.graph_objects as go  # for data visualization
import graphviz  # for plotting decision tree graphs

path = settings.MEDIA_ROOT + "//" + "heart_obecity.csv"
df = pd.read_csv(path)
X = df.iloc[:, :-1].values  # indipendent variable
y = df.iloc[:, -1].values  # Dependent variable


# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)


def fitting(X, y, criterion, splitter, mdepth, clweight, minleaf):
    # Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit the model
    model = tree.DecisionTreeClassifier(criterion=criterion,
                                        splitter=splitter,
                                        max_depth=mdepth,
                                        class_weight=clweight,
                                        min_samples_leaf=minleaf,
                                        random_state=0,
                                        )
    clf = model.fit(X_train, y_train)

    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    # Tree summary and model evaluation metrics
    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of leaves: ', clf.tree_.n_leaves)
    print('No. of features: ', clf.n_features_)
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

    # Return relevant data for chart plotting
    return X_train, X_test, y_train, y_test, clf


def start_process_cart():
    X_train, X_test, y_train, y_test, clf = fitting(X, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=1000)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    print("Sensitivity: ", sensitivity)
    specificity = tn / (tn + fp)
    print("Specificity: ", specificity)
    precision = tp/(tp+fp)
    print("Precisions: ",precision)
    f1_score = metrics.f1_score(y_test, y_pred)
    print("F1 Score: ",f1_score)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    print("Roc Auc Curve:", roc_auc)

    rslt_dict = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precisions": precision,
        "f1_score": f1_score,
        "roc_auc": roc_auc
    }
    return rslt_dict


def start_process_gbdt():
    X_train, X_test, y_train, y_test, clf = fitting(X, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=1000)
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from sklearn import metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy :", accuracy)
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    print("Sensitivity: ", sensitivity)
    specificity = tn / (tn + fp)
    print("Specificity: ", specificity)
    precision = tp/(tp+fp)
    print("Precisions: ",precision)
    f1_score = metrics.f1_score(y_test, y_pred)
    print("F1 Score: ",f1_score)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    print("Roc Auc Curve:", roc_auc)

    rslt_dict = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precisions": precision,
        "f1_score": f1_score,
        "roc_auc": roc_auc
    }
    return rslt_dict
