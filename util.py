import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean, stdev


def render_boxplot(trained, untrained, filename, labels, title=""):
    plt.title(title)
    plt.boxplot([trained, untrained],patch_artist=True,labels=labels)
    plt.savefig(f"img/{filename}.png")
    plt.close()

def render_barplot(x, y, filename, title=""):
    plt.bar(x, y)
    plt.title(title)
    plt.savefig(f"img/{filename}.png")
    plt.close()

def calculate_cohends_d(dist_1, dist_2):
    return (mean(dist_1) - mean(dist_2)) / (sqrt((stdev(dist_1) ** 2 + stdev(dist_2) ** 2) / 2))
