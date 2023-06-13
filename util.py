import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import sqrt
from statistics import mean, stdev

# Calculate Z-Score
def normalize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    z_scores = (scores - mean) / std
    return z_scores

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

def render_comparison_histogram(a,b,a_name,b_name,x_label,filename):
    #bin_min = min(min(a),min(b))
    #bin_max = max(max(a),max(b))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(a, kde=True, stat="count")
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(a_name)

    plt.subplot(1, 2, 2)
    sns.histplot(b, kde=True, stat="count")
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(b_name)

    plt.tight_layout()
    plt.savefig(f"img/{filename}.png")
    plt.close()

def calculate_cohends_d(dist_1, dist_2):
    return (mean(dist_1) - mean(dist_2)) / (sqrt((stdev(dist_1) ** 2 + stdev(dist_2) ** 2) / 2))
