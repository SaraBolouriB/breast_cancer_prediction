from read_dataset import *
from plot import plot


attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def main():
    dataset_address = "../dataset/breast-cancer-wisconsin.csv"
    dataset = read_dataset(address=dataset_address)

    # missing_value = dataset["Bare nuclei"].value_counts()['?']    #16
    # rep = dataset["Bare nuclei"].replace("?", 1)

    # maxs = maximum(db=dataset, columns=attributes)
    # mins = minimum(db=dataset, columns=attributes)
    # modes = mode(db=dataset, columns=attributes)
    # medians = median(db=dataset, columns=attributes)
    # averages = avg(db=dataset, columns=attributes)

    # bar_plot(db=dataset, column="Class")





if __name__ == "__main__":
    main()