from sklearn.naive_bayes import GaussianNB
from data import split_dataset, transform
from performance_metrics import accuracy, precision, recall, specificity, kappa, f_measure, mcc
from performance_metrics import confusionMatrix



def naive_bayes(dataset):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(dataset=dataset,
                                                                             test_size=0.20)
    
    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    NBClassifier.fit(features_train, labels_train)          #Training step
    labels_pred  =  NBClassifier.predict(features_test)      #Testing step

    ac = accuracy(labels_test=labels_test, labels_pred=labels_pred)
    kp = kappa(labels_test=labels_test, labels_pred=labels_pred)
    ps = precision(labels_test=labels_test, labels_pred=labels_pred)
    rc = recall(labels_test=labels_test, labels_pred=labels_pred)
    fm = f_measure(labels_test=labels_test, labels_pred=labels_pred)
    mc = mcc(labels_test=labels_test, labels_pred=labels_pred)
    
    print("NAIVE BAYES -----------------------" + 
          "\nAccuracy: " , ac,
          "\nKappa statistics: ", kp,
          "\nPrecision: ", ps,
          "\nrecall: ", rc,
          "\nF_measure: ", fm,
          "\nMCC: ", mc,
          "\n-----------------------------------")