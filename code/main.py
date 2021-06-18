from data import *
from plot import bar_plot, char_comparing
from naive_bayes import naive_bayes
from randomForest import random_forest
from svm import svm
from multilayer_perceptron import mlp
from cross_validate import kfold_cv_table
from performance_metrics import perf_metr_table

attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def add_to_dict(*args, table):
    if len(args) == 8:
        table['Accuracy'].append(float(args[0]))
        table['Kappa statistics'].append(float(args[1]))
        table['Precision'].append(float(args[2]))
        table['Recall'].append(float(args[3]))
        table['F_measure'].append(float(args[4]))
        table['MCC'].append(float(args[5]))
        table['ROC'].append(float(args[6]))
        table['PRC'].append(float(args[7]))
    elif len(args) == 3:
        table['Accuracy'].append(float(args[0]))
        table['Sensitivity'].append(float(args[1]))
        table['Specificity'].append(float(args[2]))
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
                     
    ### DRAW BAR PLOT FOR EACH ATTRIBUTE -------------------------------------------------------------
    # bar_plot(db=dataset, column="Class")

    table = {
        'Accuracy':[],
        'Kappa statistics':[],
        'Precision':[],
        'Recall':[],
        'F_measure':[],
        'MCC':[],
        'ROC':[],
        'PRC':[]
    }
    plot_table = {
        'Accuracy':[],
        'Sensitivity':[],
        'Specificity':[]
    }
    ### NAIVE BAYES CLASSIFICATION ALGORITHM ---------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, ss, ps = naive_bayes(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, table=table)
    plot_table = add_to_dict(ac, ss, ps, table=plot_table)

    ### RANDOM FOREST CLASSIFICATION ALGORITHM -------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, ss, ps = random_forest(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, table=table)
    plot_table = add_to_dict(ac, ss, ps, table=plot_table)

    ### SVM CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, ss, ps = svm(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, table=table)
    plot_table = add_to_dict(ac, ss, ps, table=plot_table)

    ### MLP CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, ss, ps = mlp(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, table=table)
    plot_table = add_to_dict(ac, ss, ps, table=plot_table)

    ### TABLE OF PERFORMANCE METRICS
    perf_metr_table(table=table, index=['NAIVE BAYES','RANDOM FOREST', 'SVM', 'MLP'])
    
    ### K-FOLD CROSS VALIDATE ON NAIVE BAYES ---------------------------------------------------------
    kfold_cv_table(dataset)

    char_comparing(data=plot_table, algorithm=['NB','RF', 'SVM', 'MLP'], name="Comparsion 1")
    # char_comparing(data=kfold_table, algorithm=['5-Fold','10-Fold', '15-Fold', '66.6 split', '85.5 split'], name="Comparsion 2")
    
if __name__ == "__main__":
    main()



