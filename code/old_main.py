from data import *
from plot import bar_plot, char_comparing
from naive_bayes import naive_bayes
from randomForest import random_forest
from svm import svm
from multilayer_perceptron import mlp
from cross_validate import kfold_cv_NB
from performance_metrics import perf_metr_table

attributes = ["Clump-thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare nuclei", "Bland chromatin", "Normal Nucleoli",
              "Mitoses", "Class"]

def add_to_dict(*args, table):
    if len(args) == 9:
        table['Accuracy'].append(float(args[0]))
        table['Kappa statistics'].append(float(args[1]))
        table['Precision'].append(float(args[2]))
        table['Recall'].append(float(args[3]))
        table['F_measure'].append(float(args[4]))
        table['MCC'].append(float(args[5]))
        table['ROC'].append(float(args[6]))
        table['PRC'].append(float(args[7]))
        table['Specificity'].append(float(args[8]))
    elif len(args) == 3:
        table['Accuracy'].append(float(args[0]))
        table['Sensitivity'].append(float(args[1]))
        table['Specificity'].append(float(args[2]))
    return table

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
    table = {
        'Accuracy':[],
        'Kappa statistics':[],
        'Precision':[],
        'Recall':[],
        'F_measure':[],
        'MCC':[],
        'ROC':[],
        'PRC':[],
        'Specificity':[]
    }
    plot_table = {
        'Accuracy':[],
        'Sensitivity':[],
        'Specificity':[]
    }
    ### NAIVE BAYES CLASSIFICATION ALGORITHM ---------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, sp = naive_bayes(dataset=dataset, test_size=0.20)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, sp, table=table)
    plot_table = add_to_dict(ac, rc, sp, table=plot_table)
    print(ac, kp, ps, rc, fm, mc, ra, pa, sp)
    ### RANDOM FOREST CLASSIFICATION ALGORITHM -------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, sp = random_forest(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, sp, table=table)
    plot_table = add_to_dict(ac, rc, sp, table=plot_table)

    ### SVM CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, sp = svm(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, sp, table=table)
    plot_table = add_to_dict(ac, rc, sp, table=plot_table)

    ### MLP CLASSIFICATION ALGORITHM -----------------------------------------------------------------
    ac, kp, ps, rc, fm, mc, ra, pa, sp = mlp(dataset=dataset)
    table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, sp, table=table)
    plot_table = add_to_dict(ac, rc, sp, table=plot_table)

    ### J48 CLASSIFICATION ---------------------------------------------------------------------------
    table = add_to_dict(0.928, 0.838, 0.930, 0.929, 0.929, 0.839, 0.975, 0.955, 0.924, table=table)
    plot_table = add_to_dict(0.928, 0.929, 0.924, table=plot_table)

    return table, plot_table

def dropping(dataset):
    attributes.remove('Class')
    drop_table = {
        'Accuracy':[],
        'Kappa statistics':[],
        'Precision':[],
        'Recall':[],
        'F_measure':[],
        'MCC':[],
        'ROC':[],
        'PRC':[],
        'Specificity':[]
    }
    for attr in attributes:
        new_dataset = drop_attr(dataset=dataset, attr=attr)
        kfold_cv_table(new_dataset)
        ac, kp, ps, rc, fm, mc, ra, pa, sp = naive_bayes(dataset=new_dataset, test_size=0.334)
        drop_table = add_to_dict(ac, kp, ps, rc, fm, mc, ra, pa, sp, table=drop_table)
        
    perf_metr_table(table=drop_table, index=attributes)
    attributes.append("Class")

def main():
    ### READ DATABASE --------------------------------------------------------------------------------
    dataset_address = "../dataset/breast-cancer-wisconsin.csv"
    dataset = read_dataset(address=dataset_address)

    ### PRE-PROCESSING -------------------------------------------------------------------------------
    preprocessing(dataset=dataset)
    
    ### IMPLEMENTATION -------------------------------------------------------------------------------
    table, plot_table = implementation(dataset=dataset)

    ### CHARTS AND TABLES ----------------------------------------------------------------------------
    ## TABLE OF PERFORMANCE METRICS ------------------------------------------------------------------
    perf_metr_table(table=table, index=['NAIVE BAYES','RANDOM FOREST', 'SVM', 'MLP', 'J48'])
    
    ## K-FOLD CROSS VALIDATE ON NAIVE BAYES ----------------------------------------------------------
    kfold_table = kfold_cv_NB(dataset)
    
    ## COMPARSION CHART ------------------------------------------------------------------------------
    char_comparing(data=plot_table, algorithm=['NB','RF', 'SVM', 'MLP', 'J48'], name="Comparsion 1")
    char_comparing(data=kfold_table, algorithm=['5-Fold','10-Fold', '15-Fold', '66.6 split', '85.5 split'], name="Comparsion 2")
    
    ### DROP EACH ATTRIBUTES -------------------------------------------------------------------------
    dropping(dataset=dataset)

if __name__ == "__main__":
    main()



