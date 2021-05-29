from read_dataset import read_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def naive_bayes(dataset_address):
    NBClassifier = GaussianNB()
    features_train, features_test, labels_train, labels_test = split_dataset(dataset_address=dataset_address)

    NBClassifier.fit(features_train, labels_train)          #Training step
    label_pred  =  NBClassifier.predict(features_test)      #Testing step

    ac = accuracy_score(labels_test,label_pred)
    print(ac)

def split_dataset(dataset_address):
    dataset = read_dataset(address=dataset_address)
    X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.20, 
                                                        random_state = 0)

    return X_train, X_test, y_train, y_test


naive_bayes("../dataset/breast-cancer-wisconsin.csv")