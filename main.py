"""
Ref: https://towardsdatascience.com/how-to-tune-hyperparameters-of-machine-learning-models-a82589d48fc8
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import plotly.graph_objects as go

from warnings import filterwarnings
filterwarnings('ignore')


def plot_grid_search(cv_results, name_param_1, name_param_2):
    df_params = pd.DataFrame(cv_results["params"])
    df_scores = pd.DataFrame(cv_results["mean_test_score"], columns=["Accuracy"])
    grid_results = pd.concat([df_params, df_scores], axis=1)

    grid_param_1 = df_params.columns[0]
    grid_param_2 = df_params.columns[1]

    # reshape data by pivoting them into an m ⨯ n matrix
    # where rows and columns correspond to the the first and second hyperparameter respectively
    grid_contour = grid_results.groupby([grid_param_1, grid_param_2]).mean()
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = [grid_param_1, grid_param_2, 'Accuracy']
    grid_pivot = grid_reset.pivot(grid_param_1, grid_param_2)

    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values

    # X and Y axes labels
    layout = go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text=name_param_2)
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=name_param_1)
        ))

    # Making the 3D Contour Plot
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout)

    fig.update_layout(title='Hyperparameter tuning',
                      scene=dict(
                          xaxis_title=name_param_2,
                          yaxis_title=name_param_1,
                          zaxis_title='Accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()


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
    model = SVC(kernel="rbf")

    tolerance = [0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    C = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    grid = dict(tol=tolerance, C=C)

    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
                              cv=skf, scoring="accuracy", return_train_score=True)

    searchResults = gridSearch.fit(X_train, y_train)
    # extract the best model and evaluate it
    print("[INFO] evaluating...")
    bestModel = searchResults.best_estimator_
    print("Best parameters:", searchResults.best_params_)
    print("Test accuracy: {:.3f}".format(bestModel.score(X_test, y_test)))

    plot_grid_search(cv_results=searchResults.cv_results_, name_param_1='C', name_param_2='tolerance')