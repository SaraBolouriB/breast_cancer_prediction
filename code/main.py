from data import *
from plot import bar_plot
from naive_bayes import naive_bayes
from randomForest import random_forest
from svm import svm
from multilayer_perceptron import mlp
from cross_validate import kfold_cv_table
from performance_metrics import perf_metr_table

attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def add_to_dict(ac, kp, ps, rc, fm, mc, table):
    table['Accuracy'].append(ac)
    table['Kappa statistics'].append(kp)
    table['Precision'].append(ps)
    table['Recall'].append(rc)
    table['F_measure'].append(fm)
    table['MCC'].append(mc)
    return table



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
                     
    table = {
        'Accuracy':[],
        'Kappa statistics':[],
        'Precision':[],
        'Recall':[],
        'F_measure':[],
        'MCC':[] 
    }
    ### DRAW BAR PLOT FOR EACH ATTRIBUTE -------------------------------------------------------------
    # bar_plot(db=dataset, column="Class")

    ### NAIVE BAYES CLASSIFICATION ALGORITHM ---------------------------------------------------------
    ac, kp, ps, rc, fm, mc = naive_bayes(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, table)
    
    ### RANDOM FOREST CLASSIFICATION ALGORITHM -------------------------------------------------------
    ac, kp, ps, rc, fm, mc = random_forest(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, table)
    
    ### SVM CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc = svm(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, table)
    
    ### MLP CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc = mlp(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, table)
    
    ### TABLE OF PERFORMANCE METRICS
    perf_metr_table(table=table, index=['NAIVE BAYES','RANDOM FOREST', 'SVM', 'MLP'])
    
    ### K-FOLD CROSS VALIDATE ON NAIVE BAYES ---------------------------------------------------------
    kfold_cv_table(dataset)
if __name__ == "__main__":
    main()



