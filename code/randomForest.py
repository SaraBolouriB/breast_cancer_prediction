from data import transform, split_dataset
from sklearn.ensemble import RandomForestClassifier
from performance_metrics import performance_measurement, accuracy

def random_forest(dataset):
    features_train, features_test, labels_train, labels_test = split_dataset(dataset=dataset,
                                                                             test_size=0.20,
                                                                             random_state=51)

    features_train, features_test = transform(X_train=features_train, X_test=features_test)

    regressor = RandomForestClassifier(n_estimators=100)
    regressor.fit(features_train, labels_train)
    labels_pred = regressor.predict(features_test)

    performance_measurement(labels_test=labels_test,
                            labels_pred=labels_pred,
                            algorithm_name="RANDOM FOREST")