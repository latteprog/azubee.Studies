import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind, shapiro, wilcoxon, mannwhitneyu
from math import sqrt
from statistics import mean, stdev

# Calculate Z-Score
def normalize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    z_scores = (scores - mean) / std
    return z_scores

def render_boxplot(trained, untrained, filename, labels, x_axis = "", y_axis = "", ylim = (-1.1, 1.1), title=""):
    df = pd.DataFrame({ "val": trained + untrained, "group": [labels[0][:5]] * len(trained) + [labels[1][:5]] * len(untrained) })

    with sns.color_palette("colorblind"):
        plt.clf()
        plt.figure(figsize = (4.854, 3), dpi = 300)
        sns.boxplot(data = df, x = "group", y = "val", hue = "group", order = [labels[0][:5], labels[1][:5]], width = 0.5)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.xticks([0, 1], labels)
        plt.ylim(ylim)
        plt.legend([],[], frameon=False)
        plt.tight_layout()
        print()
        plt.savefig(f"img/{filename}.png", dpi = 300)
        plt.close()

    # plt.title(title)
    # plt.boxplot([trained, untrained],patch_artist=True,labels=labels)
    # plt.savefig(f"img/{filename}.png")
    # plt.close()

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
    
def plot_pre_post(df, filename, title):
    plt.figure(figsize=(10, 6))

    # We set the width of a bar and its positions
    width = 0.3
    r1 = range(len(df))
    r2 = [x + width for x in r1]

    plt.bar(r1, df['PretestCorrectRel'], color='b', width=width, edgecolor='grey', label='PretestCorrectRel')
    plt.bar(r2, df['PosttestCorrectRel'], color='r', width=width, edgecolor='grey', label='PosttestCorrectRel')

    # Adding xticks
    plt.xlabel('Exercise Skill', fontweight='bold')
    plt.ylabel('Relative Score (in %)', fontweight='bold')
    plt.title(title)
    plt.xticks([r + width / 2 for r in range(len(df))], df['ExerciseSkill'])

    plt.legend()
    plt.savefig(f"img/{filename}.png")
    plt.close()

def calculate_cohends_d(dist_1, dist_2):
    n1, n2 = len(dist_1), len(dist_2)
    s1, s2 = stdev(dist_1), stdev(dist_2)
    x1, x2 = mean(dist_1), mean(dist_2)

    pooled_std = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (x1 - x2) / pooled_std

def perform_test(a, b, a_name, b_name, x_label, filename, is_related, is_graph_norm, norm_val=0.05, alternative='greater'):
    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=a,
        b=b,
        a_name=a_name,b_name=b_name, x_label=x_label, filename=filename)
    
    # B) Analytical Evaluation (Distribution)
    _, norm_p_a = shapiro(a)
    _, norm_p_b = shapiro(b)

    # C) Hypothesis Test
    if is_graph_norm and norm_p_a >= norm_val and norm_p_b >= norm_val:
        if is_related:
            print(f"Using a paired t-test as normality assumptions are met for {filename} with {norm_p_a} and {norm_p_b}.")
            t_test_result = ttest_rel(
                a,
                b,
                alternative=alternative
            )
        else:
            print(f"Using a t-test as normality assumptions are met for {filename} with {norm_p_a} and {norm_p_b}.")
            t_test_result = ttest_ind(
                a,
                b,
                alternative=alternative
            )
    else: 
        if is_related:
            print(f"Using a wilcoxon test as normality assumptions are not met for {filename} with {is_graph_norm} and {norm_p_a} and {norm_p_b}.")
            t_test_result = wilcoxon(
                a, 
                b,
                alternative=alternative,
                # https://www.tandfonline.com/doi/abs/10.1080/01621459.1959.10501526
                zero_method='pratt'
            )
        else:
            print(f"Using a mannwhitneyu test as normality assumptions are not met for {filename} with {is_graph_norm} and {norm_p_a} and {norm_p_b}.")
            t_test_result = mannwhitneyu(
                a, 
                b,
                alternative=alternative
            )

    return t_test_result