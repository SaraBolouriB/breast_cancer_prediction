from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from read_dataset import read_dataset

def svm(dataset):
    features_train, features_test, labels_train, labels_test = split_dataset(dataset)
    SVMClassifier = SVC(kernel='non-linear')
    SVMClassifier.fit(features_train, labels_train)

    labels_pred = SVMClassifier.predict(features_test)

    ac = accuracy_score(labels_test,labels_pred)
    print(ac)


def split_dataset(db):
    features = db.drop('Class', axis=1)
    labels = db['Class']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20)
    return X_train, X_test, y_train, y_test

dataset_address = "../dataset/breast-cancer-wisconsin.csv"
dataset = read_dataset(address=dataset_address)
svm(dataset)