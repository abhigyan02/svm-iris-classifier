# Iris SVM Classification

This repository contains a Python script that performs Support Vector Machine (SVM) classification on the Iris dataset using scikit-learn. The code utilizes grid search to find the best hyperparameters for the SVM model and evaluates its performance using a confusion matrix and accuracy score.

# Description
The script begins by importing necessary modules and loading the Iris dataset using scikit-learn.

It then splits the dataset into training and testing sets using the train_test_split function.

An SVM model is created using the svm.SVC() class.

A parameter grid is defined for grid search, specifying different values for C, gamma, and kernel.

Grid search is performed using GridSearchCV to find the best hyperparameters for the SVM model.

The best estimator found by grid search is printed.

Predictions are made on the test set using the best model, and a confusion matrix is computed using confusion_matrix.

The confusion matrix is visualized using ConfusionMatrixDisplay and matplotlib.

Finally, the accuracy score is computed using accuracy_score and printed.
# Result
The script outputs the best estimator found by grid search, the confusion matrix, and the accuracy score for the SVM model on the Iris dataset.

![image](https://github.com/abhigyan02/svm-iris-classifier/assets/75851981/307d1aa4-9ca2-4f01-acc3-695d740590a6)

![image](https://github.com/abhigyan02/svm-iris-classifier/assets/75851981/09e61dae-449c-4575-bf19-d91ad143460d)

