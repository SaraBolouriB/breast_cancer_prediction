import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

def bar_plot(db, column):
    data = db[column].value_counts().to_dict()
    names = list(data.keys())
    values = list(data.values())
    plt.bar(range(len(data)), values, tick_label=names)
    plt.xlabel(column)
    plt.ylabel("Values")
    plt.savefig("./plots/"+ column +".png")

def char_comparing(data, algorithm, name):
    plotdata = pd.DataFrame(data, index=algorithm)
    plotdata.plot(kind="bar")
    plt.title("Comparison between classifiers")
    plt.xlabel("Value")
    plt.ylabel("Classifiers")
    plt.savefig("./plots/"+ name +".png")
