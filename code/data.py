import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def read_dataset(address):
    file = pd.read_csv(address)
    return file

def avg(db, columns):
    averages = {}
    for column in columns:
        averages[column] = db[column].mean()
    return averages

def maximum(db, columns):
    maxs = {}
    for column in columns:
        maxs[column] = db[column].max()
    return maxs

def minimum(db, columns):
    mins = {}
    for column in columns:
        mins[column] = db[column].min()
    return mins

def mode(db, columns):
    modes = {}
    for column in columns:
        modes[column] = db[column].mode()
    return modes

def median(db, columns):
    medians = {}
    for column in columns:
        medians[column] = db[column].median()
    return medians

def transform(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def split_dataset(dataset, test_size, random_state):
    X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = test_size, 
                                                        random_state = random_state)

    return X_train, X_test, y_train, y_test