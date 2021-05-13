from read_dataset import *
from plot import bar_plot


attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def main():
    ### READ DATABASE --------------------------------------------------------------------------------
    dataset_address = "../dataset/breast-cancer-wisconsin.csv"
    dataset = read_dataset(address=dataset_address)

    ### FINDING MISSING VALUE AND REPLACE IT WITH MODE -----------------------------------------------
    missing_value = dataset["Bare nuclei"].value_counts()['?']    #16
    rep = dataset["Bare nuclei"].replace("?", 1)

    ### CALCULATE MAX, MIN, MODE, MEDIAN, AVERAGE FOR EACH ATTRIBUTE ---------------------------------
    maxs = maximum(db=dataset, columns=attributes)
    mins = minimum(db=dataset, columns=attributes)
    modes = mode(db=dataset, columns=attributes)
    medians = median(db=dataset, columns=attributes)
    averages = avg(db=dataset, columns=attributes)

    for attr in attributes:
        print(attr + ":\n Min:" , mins[attr] ,
                     "\n Max:"  , maxs[attr] ,
                     "\n Mode:" , modes[attr][0],
                     "\n Median:" , medians[attr],
                     "\n Average:" , averages[attr])
                     
    ### DRAW BAR PLOT FOR EACH ATTRIBUTE -------------------------------------------------------------
    bar_plot(db=dataset, column="Class")

if __name__ == "__main__":
    main()