import pandas as pd

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
