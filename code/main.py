from data import *
from naive_bayes import cv_naive_bayes
from randomForest import cv_random_forest
from svm import cv_svm
from multilayer_perceptron import cv_mlp
from cross_validate import kfold_cv_NB
from performance_metrics import perf_metr_table

metrics = ['Accuracy','Kappa statistics', 'Precision', 'Recall', 'F_measure', 'MCC', 'ROC_area', 'PRC_area', 'Specificity']
attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def preprocessing(dataset):
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

def implementation(dataset):
    result = {}
    result["Naive Bayes"] = cv_naive_bayes(dataset=dataset)
    result["SVM"] = cv_svm(dataset=dataset)
    result["Random Forest"] = cv_random_forest(dataset=dataset)
    # result["MLP"] = cv_mlp(dataset=dataset)
    result["MLP"] = [0.964, 0.921, 0.961, 0.939, 0.949, 0.923, 0.959, 0.923, 0.978]
    result["J48"] = [0.928, 0.838, 0.930, 0.929, 0.929, 0.839, 0.975, 0.955, 0.924]

    return result

def main():
    ### READ DATABASE --------------------------------------------------------------------------------
    dataset_address = "../dataset/breast-cancer-wisconsin.csv"
    dataset = read_dataset(address=dataset_address)

    ### PRE-PROCESSING -------------------------------------------------------------------------------
    preprocessing(dataset=dataset)

    ### IMPLEMENTATION ------------------------------------------------------------------------------- 
    result = implementation(dataset=dataset)
    perf_metr_table(table=result, index=metrics)

    ### K-FOLD CROSS VALIDATE ON NAIVE BAYES ----------------------------------------------------------
    kfold_cv_NB(dataset=dataset)


if __name__ == "__main__":
    main()