import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from warnings import filterwarnings
filterwarnings('ignore')


if __name__ == '__main__':

    # Load default data with corresponding features name
    dataset = pd.read_csv('data/breast-cancer-wisconsin.data',
                          names=[
                              "Sample_code_number",
                              "Clump_Thickness",
                              "Uniformity_of_Cell_Size",
                              "Uniformity_of_Cell_Shape",
                              "Marginal_Adhesion",
                              "Single_Epithelial_Cell_Size",
                              "Bare_Nuclei",
                              "Bland_Chromatin",
                              "Normal_Nucleoli",
                              "Mitoses",
                              "Class"
                          ],
                          sep=',',
                          na_values='?')

    dataset = dataset.drop("Sample_code_number", axis=1)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    # class imbalance dataset => use StratifiedKFold below
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    imputer = IterativeImputer()

    # avoid data leakage (only fit the imputer with the training data)
    imputer.fit(X_train)

    # imputed X_train and X_test
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    # round the imputed values
    X_train = X_train.round(0)
    X_test = X_test.round(0)

    # convert type to ndarray
    y_train = y_train.values
    y_test = y_test.values

    # prepare the cross-validation procedure
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # create model
    model = MLPClassifier(hidden_layer_sizes=(5, 5, 5,), activation='relu')
    scores = []

    for train_index, val_index in skf.split(X_train, y_train):
        # print("TRAIN:", train_index, "TEST:", val_index)
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

    # report performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    print(np.isnan(np.sum(X_test)))