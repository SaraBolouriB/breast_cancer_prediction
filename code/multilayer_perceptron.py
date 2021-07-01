from data import transform, split_dataset
from sklearn.neural_network import MLPClassifier
from performance_metrics import performance_measurement, performance_measurement_cv

def mlp(dataset):
    mlpClassifier = MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=1000,
        activation='relu', 
        solver='adam', 
        random_state=1
    )
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset, 
        test_size=0.20,
        random_state=0
    )
    
    mlpClassifier.fit(features_train, labels_train)     #Training step
    labels_pred = mlpClassifier.predict(features_test)  #Testing step

    ac, kp, ps, rc, fm, mc, ra, pa, sp = performance_measurement(
                                            labels_test=labels_test, 
                                            labels_pred=labels_pred,
                                            algorithm_name="MLP"
                                        )
    return ac, kp, ps, rc, fm, mc, ra, pa, sp

def cv_mlp(dataset):
    mlpClassifier = MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=1000,
        activation='relu', 
        solver='adam', 
        random_state=1
    )
    features_train, features_test, labels_train, labels_test = split_dataset(
        dataset=dataset, 
        test_size=0.20,
        random_state=0
    )
    result = performance_measurement_cv(algorithm=mlpClassifier, features_train=features_train, labels_train=labels_train)
                                 
    return result                                                      
