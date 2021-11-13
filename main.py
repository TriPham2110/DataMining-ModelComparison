import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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

    print(X_test)
    print(np.isnan(np.sum(X_test)))