"""
Ref: https://towardsdatascience.com/how-to-tune-hyperparameters-of-machine-learning-models-a82589d48fc8
"""

import random
import time
from warnings import filterwarnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict

filterwarnings('ignore')


def plot_grid_search(cv_results, name_param_1, name_param_2, model_name):
    df_params = pd.DataFrame(cv_results["params"])
    if df_params[name_param_1].dtypes == object:
        df_params[name_param_1] = df_params[name_param_1].apply(str)
    if df_params[name_param_2].dtypes == object:
        df_params[name_param_2] = df_params[name_param_2].apply(str)
    df_scores = pd.DataFrame(np.round(cv_results["mean_test_score"], decimals=3), columns=["Accuracy"])
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
    fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)],
                    layout=layout)

    fig.update_layout(title='Hyperparameter tuning for ' + model_name,
                      scene=OrderedDict(
                          xaxis_title=name_param_2,
                          yaxis_title=name_param_1,
                          zaxis_title='Train Accuracy'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()


def grid_search(model, grid, model_name):
    # prepare the cross-validation procedure
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    gridSearch = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
                              cv=skf, scoring="accuracy", return_train_score=True, verbose=1)
    searchResults = gridSearch.fit(X_train, y_train)

    # extract the best model and evaluate it
    print("[INFO] evaluating...")
    bestModel = searchResults.best_estimator_
    print("Best considered parameters:", searchResults.best_params_)

    plot_grid_search(cv_results=searchResults.cv_results_,
                     name_param_1=list(grid.keys())[0],
                     name_param_2=list(grid.keys())[1],
                     model_name=model_name)

    return bestModel


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

    print(y_test.value_counts())

    imputer = IterativeImputer()

    # avoid data leakage (only fit the imputer with the training data)
    imputer.fit(X_train)

    # imputed X_train and X_test
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    # round the imputed values
    X_train = X_train.round(0)
    X_test = X_test.round(0)

    # standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # convert type to ndarray
    y_train = y_train.values
    y_test = y_test.values

    # performing grid search cross validation for k-nearest neighbors
    print("Performing grid search cross validation for k-nearest neighbors")
    start_knn = time.time()
    knn = KNeighborsClassifier(n_jobs=-1)
    n_neighbors = np.arange(1, 51, 1)
    p = np.arange(1, 51, 1)
    best_tuned_knn = grid_search(model=knn,
                                 grid=OrderedDict(n_neighbors=n_neighbors, p=p),
                                 model_name="k-nearest neighbors")
    end_knn = time.time()

    # performing grid search cross validation for decision tree
    print("Performing grid search cross validation for decision tree")
    start_dt = time.time()
    dt = DecisionTreeClassifier(random_state=42)
    max_depth = np.arange(1, 101, 1)
    min_samples_leaf = np.arange(1, 101, 1)
    best_tuned_dt = grid_search(model=dt,
                                grid=OrderedDict(max_depth=max_depth, min_samples_leaf=min_samples_leaf),
                                model_name="decision tree")
    end_dt = time.time()

    # performing grid search cross validation for random forest
    print("Performing grid search cross validation for random forest")
    start_rf = time.time()
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    max_depth = np.arange(1, 101, 1)
    n_estimators = np.arange(100, 1001, 100)
    best_tuned_rf = grid_search(model=rf,
                                grid=OrderedDict(max_depth=max_depth, n_estimators=n_estimators),
                                model_name="random forest")
    end_rf = time.time()

    # performing grid search cross validation for SVM using the polynomial kernel
    print("Performing grid search cross validation for SVM using the polynomial kernel")
    start_poly_svm = time.time()
    poly_svm = SVC(kernel="poly", random_state=42)
    degree = np.arange(2, 12, 1)
    C = np.arange(0.1, 10.1, 0.1)
    best_tuned_poly_svm = grid_search(model=poly_svm,
                                      grid=OrderedDict(C=C, degree=degree),
                                      model_name="SVM using the polynomial kernel")
    end_poly_svm = time.time()

    # performing grid search cross validation for SVM using the RBF kernel
    print("Performing grid search cross validation for SVM using the RBF kernel")
    start_rbf_svm = time.time()
    rbf_svm = SVC(kernel="rbf", random_state=42)
    gamma = np.arange(1e-5, 10.1, 0.1)
    C = np.arange(0.1, 10.1, 0.1)
    best_tuned_rbf_svm = grid_search(model=rbf_svm,
                                     grid=OrderedDict(C=C, gamma=gamma),
                                     model_name="SVM using the RBF kernel")
    end_rbf_svm = time.time()

    nn_hidden_layers = []

    for i in range(5, 31, 5):
        for j in range(5, 31, 5):
            for k in range(5, 31, 5):
                nn_hidden_layers.append((i, j, k))

    # performing grid search cross validation for Deep neural network with sigmoid activation
    print("Performing grid search cross validation for deep neural network with sigmoid activation")
    start_sigmoid_nn = time.time()
    sigmoid_nn = MLPClassifier(activation='logistic', max_iter=3000, random_state=42)
    hidden_layer_sizes = [(random.randrange(0, 100), random.randrange(0, 100), random.randrange(0, 100)) for i in range(100)]
    learning_rate_init = np.arange(1e-3, 1.1, 0.1)
    best_tuned_sigmoid_nn = grid_search(model=sigmoid_nn,
                                        grid=OrderedDict(hidden_layer_sizes=nn_hidden_layers, learning_rate_init=learning_rate_init),
                                        model_name="deep neural network with sigmoid activation")
    end_sigmoid_nn = time.time()

    # performing grid search cross validation for Deep neural network with relu activation
    print("Performing grid search cross validation for deep neural network with relu activation")
    start_relu_nn = time.time()
    relu_nn = MLPClassifier(activation='relu', max_iter=3000, random_state=42)
    hidden_layer_sizes = [(random.randrange(0, 100), random.randrange(0, 100), random.randrange(0, 100)) for i in range(100)]
    learning_rate_init = np.arange(1e-3, 1.1, 0.1)
    best_tuned_relu_nn = grid_search(model=relu_nn,
                                     grid=OrderedDict(hidden_layer_sizes=nn_hidden_layers, learning_rate_init=learning_rate_init),
                                     model_name="deep neural network with relu activation")
    end_relu_nn = time.time()

    print("\nTime taken to tune each model's hyperparameters")
    print("K-nearest neighbors:", end_knn - start_knn)
    print("Decision tree:", end_dt - start_dt)
    print("Random forest:", end_rf - start_rf)
    print("SVM using the polynomial kernel:", end_poly_svm - start_poly_svm)
    print("SVM using the RBF kernel:", end_rbf_svm - start_rbf_svm)
    print("Deep neural network with sigmoid activation:", end_sigmoid_nn - start_sigmoid_nn)
    print("Deep neural network with relu activation:", end_relu_nn - start_relu_nn)

    models = [best_tuned_knn, best_tuned_dt, best_tuned_rf, best_tuned_poly_svm, best_tuned_rbf_svm, best_tuned_sigmoid_nn, best_tuned_relu_nn]
    test_times = []
    test_scores = []

    for model in models:
        start_model_test = time.time()
        model_test_score = model.score(X_test, y_test)
        end_model_test = time.time()
        test_times.append(end_model_test - start_model_test)
        test_scores.append(model_test_score)

    print("\nTime taken for each model on final test data")
    print("K-nearest neighbors:", test_times[0])
    print("Decision tree:", test_times[1])
    print("Random forest:", test_times[2])
    print("SVM using the polynomial kernel:", test_times[3])
    print("SVM using the RBF kernel:", test_times[4])
    print("Deep neural network with sigmoid activation:", test_times[5])
    print("Deep neural network with relu activation:", test_times[6])

    print("\nTest accuracy")
    print("K-nearest neighbors: {:.3f}".format(test_scores[0]))
    print("Decision tree: {:.3f}".format(test_scores[1]))
    print("Random forest: {:.3f}".format(test_scores[2]))
    print("SVM using the polynomial kernel: {:.3f}".format(test_scores[3]))
    print("SVM using the RBF kernel: {:.3f}".format(test_scores[4]))
    print("Deep neural network with sigmoid activation: {:.3f}".format(test_scores[5]))
    print("Deep neural network with relu activation: {:.3f}".format(test_scores[6]))
