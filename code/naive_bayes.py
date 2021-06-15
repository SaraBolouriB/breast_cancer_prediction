from sklearn.naive_bayes import GaussianNB
from data import split_dataset, transform
from performance_metrics import performance_measurement



def naive_bayes(dataset):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset,
        test_size=0.20,
        random_state=51
    )
    
    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    NBClassifier.fit(features_train, labels_train)          #Training step
    labels_pred  =  NBClassifier.predict(features_test)      #Testing step

    performance_measurement(
        labels_test=labels_test, 
        labels_pred=labels_pred,
        algorithm_name="NAIVE BAYES"
    )