import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel, ttest_ind, shapiro, wilcoxon, mannwhitneyu
from math import sqrt
from statistics import mean, stdev
import os  # Added for directory existence check


# Calculate Z-Score
def normalize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    z_scores = (scores - mean) / std
    return z_scores


def render_boxplot(trained, untrained, filename, labels, title=""):
    plt.title(title)
    plt.boxplot([trained, untrained], patch_artist=True, labels=labels)
    plt.savefig(f"img/{filename}.png")
    plt.close()


def render_barplot(x, y, filename, title=""):
    plt.bar(x, y)
    plt.title(title)
    plt.savefig(f"img/{filename}.png")
    plt.close()


def render_comparison_histogram(
    a, b, a_name, b_name, x_label, filename, output_dir="img"
):
    render_comparison_histograms(
        data_list=[
            {"name": a_name, "values": a},
            {"name": b_name, "values": b},
        ],
        x_label=x_label,
        filename=filename,
        output_dir=output_dir,
    )


# Enhanced function to render comparison histograms with more customization
def render_comparison_histograms(data_list, x_label, filename, output_dir="img"):
    if not os.path.exists(
        output_dir
    ):  # Check if the output directory exists, create if not
        os.makedirs(output_dir)

    num_plots = len(data_list)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

    plt.figure(figsize=(12, 4 * num_rows))

    for i, data in enumerate(data_list, start=1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(data["values"], kde=True, stat="count")
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.title(data["name"])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()


def plot_pre_post(df, filename, title):
    plt.figure(figsize=(10, 6))

    # We set the width of a bar and its positions
    width = 0.3
    r1 = range(len(df))
    r2 = [x + width for x in r1]

    plt.bar(
        r1,
        df["PretestCorrectRel"],
        color="b",
        width=width,
        edgecolor="grey",
        label="PretestCorrectRel",
    )
    plt.bar(
        r2,
        df["PosttestCorrectRel"],
        color="r",
        width=width,
        edgecolor="grey",
        label="PosttestCorrectRel",
    )

    # Adding xticks
    plt.xlabel("Exercise Skill", fontweight="bold")
    plt.ylabel("Relative Score (in %)", fontweight="bold")
    plt.title(title)
    plt.xticks([r + width / 2 for r in range(len(df))], df["ExerciseSkill"])

    plt.legend()
    plt.savefig(f"img/{filename}.png")
    plt.close()


def calculate_cohends_d(dist_1, dist_2):
    n1, n2 = len(dist_1), len(dist_2)
    s1, s2 = stdev(dist_1), stdev(dist_2)
    x1, x2 = mean(dist_1), mean(dist_2)

    pooled_std = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (x1 - x2) / pooled_std


def perform_test(
    a,
    b,
    a_name,
    b_name,
    x_label,
    filename,
    is_related,
    is_graph_norm,
    norm_val=0.05,
    alternative="greater",
    output_dir="img",
):
    a, b = np.array(a), np.array(
        b
    )  # Ensure inputs are numpy arrays for statistical operations

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Input arrays must not be empty.")

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a, b, a_name, b_name, x_label, filename, output_dir=output_dir
    )

    # B) Analytical Evaluation (Distribution)
    _, norm_p_a = shapiro(a)
    _, norm_p_b = shapiro(b)
    print(
        f"Shapiro-Wilk test for {filename}: p-value A={norm_p_a}, p-value B={norm_p_b}"
    )

    # C) Hypothesis Test
    std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
    mean_a, mean_b = np.mean(a), np.mean(b)

    if is_graph_norm and norm_p_a >= norm_val and norm_p_b >= norm_val:
        if is_related:
            test_result = ttest_rel(a, b, alternative=alternative)
            test_name = "paired t-test"
        else:
            test_result = ttest_ind(a, b, alternative=alternative)
            test_name = "t-test"

        print(
            f"Using {test_name} as normality assumptions are met for {filename}."
            f", std_a={std_a}, std_b={std_b}, mean_a={mean_a}, mean_b={mean_b}"
        )
        print(
            f"Test result: statistic={test_result.statistic}, p-value={test_result.pvalue}"
        )
    else:
        if is_related:
            test_result = wilcoxon(
                a,
                b,
                alternative=alternative,
                # https://www.tandfonline.com/doi/abs/10.1080/01621459.1959.10501526
                zero_method="pratt",
            )
            test_name = "Wilcoxon test"
        else:
            test_result = mannwhitneyu(a, b, alternative=alternative)
            test_name = "Mann-Whitney U test"

        print(
            f"Using {test_name} as normality assumptions are not met for {filename}."
            f", std_a={std_a}, std_b={std_b}, mean_a={mean_a}, mean_b={mean_b}"
        )
        print(
            f"Test result: statistic={test_result.statistic}, p-value={test_result.pvalue}"
        )

    return test_result.statistic, test_result.pvalue, test_name
