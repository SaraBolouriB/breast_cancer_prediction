from sklearn.naive_bayes import GaussianNB
from data import split_dataset, transform
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score
import pandas as pd  


def cv_naive_bayes(dataset, rd, cv, scoring):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=0.2,
        random_state=rd
    )
    
    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    cv_results = cross_val_score(
        NBClassifier, 
        features_train, 
        labels_train,
        cv = cv,
        scoring = scoring
    )
                                 
    return cv_results.mean()


def kfold_cv_table(dataset):
    tabel = {}
    measurments = ['accuracy', 'kappa', 'precision', 'recall', 'F-measure', 'MCC', 'ROC area', 'PRC area']
    kappa = make_scorer(cohen_kappa_score)
    mcc = make_scorer(matthews_corrcoef)
    roc = make_scorer(roc_auc_score)
    prc = make_scorer(average_precision_score)
    scores = ['accuracy',kappa, 'precision', 'recall', 'f1', mcc, roc, prc]
    rd = [99, 99 , 48, 99, 99, 99, 99, 99]
    i = 0
    for score in scores:
        name = measurments[i]
        tabel[name] = []
        tabel[name].append('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=5, scoring=score))
        tabel[name].append('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=10, scoring=score))
        tabel[name].append('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=15, scoring=score))
        i += 1
    
    df = pd.DataFrame(tabel, index =['5-fold', '10-fold', '15-fold'])  
    print(df)