from sklearn.svm import SVC
from data import split_dataset, transform
from performance_metrics import performance_measurement, performance_measurement_cv


def svm(dataset):
    SVMClassifier = SVC(kernel='linear')
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=20,
        random_state=51
    )

    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    SVMClassifier.fit(features_train, labels_train)
    labels_pred = SVMClassifier.predict(features_test)

    ac, kp, ps, rc, fm, mc, ra, pa, sp = performance_measurement(
                                            labels_test=labels_test, 
                                            labels_pred=labels_pred,
                                            algorithm_name="SVM"
                                        )
    return ac, kp, ps, rc, fm, mc, ra, pa, sp


def cv_svm(dataset):
    SVMClassifier = SVC(kernel='linear')
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=0.2,
        random_state=51
    )

    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    result = performance_measurement_cv(algorithm=SVMClassifier, features_train=features_train, labels_train=labels_train)
                                 
    return result