from sklearn.naive_bayes import GaussianNB
from data import split_dataset, transform
from performance_metrics import performance_measurement, performance_measurement_cv

def naive_bayes(dataset, test_size):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=test_size,
        random_state=51
    )
    
    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    NBClassifier.fit(features_train, labels_train)           #Training step
    labels_pred  =  NBClassifier.predict(features_test)      #Testing step

    ac, kp, ps, rc, fm, mc, ra, pa, sp = performance_measurement(
                                            labels_test=labels_test, 
                                            labels_pred=labels_pred,
                                            algorithm_name="NAIVE BAYES"
                                        )
    return ac, kp, ps, rc, fm, mc, ra, pa, sp

def cv_naive_bayes(dataset):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=0.20,
        random_state=51
    )
    
    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    result = performance_measurement_cv(algorithm=NBClassifier, features_train=features_train, labels_train=labels_train)
                                 
    return result