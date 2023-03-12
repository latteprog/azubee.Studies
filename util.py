import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from statistics import mean, stdev


def render_boxplot(trained, untrained, filename, labels):
    ax = sns.boxplot(data=[trained, untrained], medianprops={'color': 'purple', 'lw': 2})
    ax.set_xticklabels(labels)
    plt.savefig(f"img/{filename}.png")
    plt.close()


def render_barplot(x, y, filename, title="", labels=None):
    ax = sns.barplot(x=x, y=y)
    ax.set(title=title)
    if labels is not None:
        ax.set_xticklabels(labels)
    plt.savefig(f"img/{filename}.png")
    plt.close()


def calculate_cohends_d(dist_1, dist_2):
    return (mean(dist_1) - mean(dist_2)) / (sqrt((stdev(dist_1) ** 2 + stdev(dist_2) ** 2) / 2))
