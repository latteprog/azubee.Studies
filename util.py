import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, shapiro, wilcoxon
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
    render_comparison_histograms(
        data_list= [
            {"name": a_name, "values": a},
            {"name": b_name, "values": b},
        ],
        x_label=x_label,
        filename=filename
    )

def render_comparison_histograms(data_list, x_label, filename):
    num_plots = len(data_list)
    num_cols = 2  # set the number of columns in the figure
    num_rows = math.ceil(num_plots / num_cols)

    plt.figure(figsize=(12, 4 * num_rows))
    
    for i, data in enumerate(data_list, start=1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(data['values'], kde=True, stat="count")
        plt.xlabel(x_label)
        plt.ylabel('Frequency')
        plt.title(data['name'])

    plt.tight_layout()
    plt.savefig(f"img/{filename}.png")
    plt.close()

def calculate_cohends_d(dist_1, dist_2):
    return (mean(dist_1) - mean(dist_2)) / (sqrt((stdev(dist_1) ** 2 + stdev(dist_2) ** 2) / 2))


def perform_test(a, b, a_name, b_name, x_label, filename, is_graph_norm, norm_val=0.05, alternative='greater'):
    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=a,
        b=b,
        a_name=a_name,b_name=b_name, x_label=x_label, filename=filename)
    
    # B) Analytical Evaluation (Distribution)
    _, norm_p_a = shapiro(a)
    _, norm_p_b = shapiro(b)

    # C) Hypothesis Test
    if is_graph_norm and norm_p_a <= norm_val and norm_p_b <= norm_val:
        print(f"Using a related t-test as normality assumptions are met for {filename} with {norm_p_a} and {norm_p_b}.")
        
        t_test_result = ttest_rel(
            a,
            b,
            alternative=alternative
        )
    else: 
        print(f"Using a wilcoxon test as normality assumptions are not met for {filename} with {is_graph_norm} and {norm_p_a} and {norm_p_b}.")

        t_test_result = wilcoxon(
            a, 
            b,
            alternative=alternative#,
            # https://www.tandfonline.com/doi/abs/10.1080/01621459.1959.10501526
            #zero_method='pratt'
        )

    return t_test_result