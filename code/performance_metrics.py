from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

float_format = '%.3f'

def accuracy(labels_test, labels_pred):
    ac = float_format % accuracy_score(labels_test,labels_pred)
    return ac

def precision(labels_test, labels_pred):
    pc = float_format % precision_score(labels_test, labels_pred)
    return pc

def recall(labels_test, labels_pred):
    rc = float_format % recall_score(labels_test, labels_pred)
    return rc

def confusionMatrix(labels_test, labels_pred):
    cm = confusion_matrix(labels_test, labels_pred)
    return cm

def specificity(labels_test, labels_pred):
    cm = confusionMatrix(labels_test=labels_test, labels_pred=labels_pred)
    TN = cm[0][0]
    FP = cm[0][1]
    return float_format % TN / (TN+FP)

def sensitivity(labels_test, labels_pred):
    cm = confusionMatrix(labels_test=labels_test, labels_pred=labels_pred)
    TP = cm[1][1]
    FN = cm[1][0]
    return float_format % TP / (TP+FN)

def f_measure(labels_test, labels_pred):
    fm = float_format % f1_score(labels_test, labels_pred)
    return fm

def kappa(labels_test, labels_pred):
    kp = float_format % cohen_kappa_score(labels_test, labels_pred)
    return kp

def mcc(labels_test, labels_pred):
    m = float_format % matthews_corrcoef(labels_test, labels_pred)
    return m

def performance_measurement(labels_test, labels_pred, algorithm_name):
    ac = accuracy(labels_test=labels_test, labels_pred=labels_pred)
    kp = kappa(labels_test=labels_test, labels_pred=labels_pred)
    ps = precision(labels_test=labels_test, labels_pred=labels_pred)
    rc = recall(labels_test=labels_test, labels_pred=labels_pred)
    fm = f_measure(labels_test=labels_test, labels_pred=labels_pred)
    mc = mcc(labels_test=labels_test, labels_pred=labels_pred)
    
    print(algorithm_name + "-----------------------" + 
          "\nAccuracy: " , ac,
          "\nKappa statistics: ", kp,
          "\nPrecision: ", ps,
          "\nrecall: ", rc,
          "\nF_measure: ", fm,
          "\nMCC: ", mc,
          "\n-----------------------------------")