import matplotlib.pyplot as plt

def bar_plot(db, column):
    data = db[column].value_counts().to_dict()
    names = list(data.keys())
    values = list(data.values())
    plt.bar(range(len(data)), values, tick_label=names)
    plt.xlabel(column)
    plt.ylabel("Values")
    plt.savefig("./plots/"+ column +".png")