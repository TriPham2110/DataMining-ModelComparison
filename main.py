import pandas as pd

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
                          sep=',')

    print(dataset)