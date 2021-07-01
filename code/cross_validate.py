from sklearn.naive_bayes import GaussianNB
from naive_bayes import naive_bayes
from data import split_dataset, transform
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from sklearn.model_selection import cross_val_score
import pandas as pd  


def cv_naive_bayes(dataset, rd, cv, scoring, test_size):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=test_size,
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

def kfold_cv_NB(dataset):
    table = {}
    table_plot = {}
    measurments = ['accuracy', 'kappa', 'precision', 'recall', 'F-measure', 'MCC', 'ROC area', 'PRC area', 'specificity']
    rd = [99, 99 , 48, 99, 99, 99, 99, 99, 99]

    kappa = make_scorer(cohen_kappa_score)
    mcc = make_scorer(matthews_corrcoef)
    roc = make_scorer(roc_auc_score)
    prc = make_scorer(average_precision_score)
    sp = make_scorer(recall_score, pos_label=0)
    scores = ['accuracy',kappa, 'precision', 'recall', 'f1', mcc, roc, prc, sp]
    
    split66 = naive_bayes(dataset=dataset, test_size=0.334)
    split85 = naive_bayes(dataset=dataset, test_size=0.145)

    i = 0
    for score in scores:
        name = measurments[i]

        five_k = float('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=5, scoring=score, test_size=0.2))
        ten_k = float('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=10, scoring=score, test_size=0.2))
        fif_k = float('%.3f' % cv_naive_bayes(dataset=dataset, rd=rd[i], cv=15, scoring=score, test_size=0.2))

        if name == 'accuracy' or name == 'recall' or name == 'specificity':
            table_plot = add_to_dic(five_k, ten_k, fif_k, split66[i], split85[i], name=name, dic=table_plot)
        
        table = add_to_dic(five_k, ten_k, fif_k, split66[i], split85[i], name=name, dic=table)
        i += 1

    df = pd.DataFrame(table, index =['5-fold', '10-fold', '15-fold', '66.6 split', '85.5 split'])  
    print(df)

    return table_plot

def add_to_dic(*args, name, dic):
    dic[name] = []
    dic[name].append(args[0])
    dic[name].append(args[1])
    dic[name].append(args[2])
    dic[name].append(args[3])
    dic[name].append(args[4])
    return dic