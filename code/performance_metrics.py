from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd  
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

float_format = '%.3f'

def accuracy(labels_test, labels_pred):
    ac = float_format % accuracy_score(labels_test,labels_pred)
    return ac

def precision(labels_test, labels_pred):
    pc = float_format % precision_score(labels_test, labels_pred)
    return pc

def recall(labels_test, labels_pred):
    # Sencitivity and Recall
    rc = float_format % recall_score(labels_test, labels_pred)
    return rc

def confusionMatrix(labels_test, labels_pred):
    cm = confusion_matrix(labels_test, labels_pred)
    return cm

def specificity(labels_test, labels_pred):
    rc = float_format % recall_score(labels_test, labels_pred, pos_label=0)
    return rc

def f_measure(labels_test, labels_pred):
    fm = float_format % f1_score(labels_test, labels_pred)
    return fm

def kappa(labels_test, labels_pred):
    kp = float_format % cohen_kappa_score(labels_test, labels_pred)
    return kp

def mcc(labels_test, labels_pred):
    m = float_format % matthews_corrcoef(labels_test, labels_pred)
    return m

def roc_area(labels_test, labels_pred):
    ra = float_format % roc_auc_score(labels_test, labels_pred)
    return ra

def prc_area(labels_test, labels_pred):
    pa = float_format % average_precision_score(labels_test, labels_pred)
    return pa

def performance_measurement(labels_test, labels_pred, algorithm_name):
    ac = accuracy(labels_test=labels_test, labels_pred=labels_pred)
    kp = kappa(labels_test=labels_test, labels_pred=labels_pred)
    ps = precision(labels_test=labels_test, labels_pred=labels_pred)
    rc = recall(labels_test=labels_test, labels_pred=labels_pred)
    fm = f_measure(labels_test=labels_test, labels_pred=labels_pred)
    mc = mcc(labels_test=labels_test, labels_pred=labels_pred)
    ra = roc_area(labels_test=labels_test, labels_pred=labels_pred)
    pa = prc_area(labels_test=labels_test, labels_pred=labels_pred)
    sp = specificity(labels_test=labels_test, labels_pred=labels_pred)

    # print(algorithm_name + "-----------------------" + 
    #       "\nAccuracy: " , ac,
    #       "\nKappa statistics: ", kp,
    #       "\nPrecision: ", ps,
    #       "\nRecall: ", rc,
    #       "\nF_measure: ", fm,
    #       "\nMCC: ", mc,
    #       "\nROC_area: ", ra,
    #       "\nPRC_area: ", pa,
    #       "\nSpecificity: ", sp,
    #       "\n-----------------------------------")
        
    return ac, kp, ps, rc, fm, mc, ra, pa, sp

def performance_measurement_cv(algorithm, features_train, labels_train):
    result = []
    kappa = make_scorer(cohen_kappa_score)
    mcc = make_scorer(matthews_corrcoef)
    roc = make_scorer(roc_auc_score)
    prc = make_scorer(average_precision_score)
    sp = make_scorer(recall_score, pos_label=0)
    scores = ['accuracy',kappa, 'precision', 'recall', 'f1', mcc, roc, prc, sp]

    for score in scores:
        result.append(float(float_format % cross_val_score(
                algorithm, 
                features_train, 
                labels_train,
                cv = 10,
                scoring = score
            ).mean())
        )
    return result

def perf_metr_table(table, index, type):
    if type == 1:
        df = pd.DataFrame(table, index=index)  
    elif type == 2:
        df = pd.DataFrame.from_dict(
            table,
            orient='index',
            columns=index
        )
    print(df)