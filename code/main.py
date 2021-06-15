from data import *
from plot import bar_plot
from naive_bayes import naive_bayes
from randomForest import random_forest
from svm import svm
from multilayer_perceptron import mlp
from cross_validate import kfold_cv_table, cv_naive_bayes
from sklearn.metrics import make_scorer, cohen_kappa_score, matthews_corrcoef

attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def main():
    ### READ DATABASE --------------------------------------------------------------------------------
    dataset_address = "../dataset/breast-cancer-wisconsin.csv"
    dataset = read_dataset(address=dataset_address)

    ### FINDING MISSING VALUE AND REPLACE IT WITH MODE -----------------------------------------------
    # missing_value = dataset["Bare nuclei"].value_counts()['?']    #16
    # rep = dataset["Bare nuclei"].replace("?", 1)

    ### CALCULATE MAX, MIN, MODE, MEDIAN, AVERAGE FOR EACH ATTRIBUTE ---------------------------------
    # maxs = maximum(db=dataset, columns=attributes)
    # mins = minimum(db=dataset, columns=attributes)
    # modes = mode(db=dataset, columns=attributes)
    # medians = median(db=dataset, columns=attributes)
    # averages = avg(db=dataset, columns=attributes)

    # for attr in attributes:
    #     print(attr + ":\n Min:" , mins[attr] ,
    #                  "\n Max:"  , maxs[attr] ,
    #                  "\n Mode:" , modes[attr][0],
    #                  "\n Median:" , medians[attr],
    #                  "\n Average:" , averages[attr])
                     
    ### DRAW BAR PLOT FOR EACH ATTRIBUTE -------------------------------------------------------------
    # bar_plot(db=dataset, column="Class")

    ### NAIVE BAYES CLASSIFICATION ALGORITHM ---------------------------------------------------------
    # naive_bayes(dataset=dataset)

    ### RANDOM FOREST CLASSIFICATION ALGORITHM -------------------------------------------------------
    # random_forest(dataset=dataset)

    ### SVM CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    # svm(dataset=dataset)

    ### MLP CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    # mlp(dataset=dataset)

    ### K-FOLD CROSS VALIDATE ON NAIVE BAYES ---------------------------------------------------------
    kfold_cv_table(dataset)
    # print('%.3f' % cv_naive_bayes(dataset=dataset, rd=99, cv=5, scoring='precision'))
    # maxx = []
    # mcc = make_scorer(matthews_corrcoef)
    # for i in range(100):
    #     maxx.append(cv_naive_bayes(dataset=dataset, rd=i, cv=10, scoring=mcc))
    # print(max(maxx))
if __name__ == "__main__":
    main()