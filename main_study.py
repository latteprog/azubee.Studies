import pandas as pd
import numpy as np
from util import (
    render_comparison_histogram,
    calculate_cohends_d,
    perform_test,
    plot_pre_post,
    render_boxplot,
)
from scipy.stats import levene, f
import matplotlib.pyplot as plt

plt.legend()
# Exercise 1 : it-network-plan-vlan
# Exercise 2 : it-network-plan-ipv4-static-routing
# Exercise 3 : it-network-plan-ipv4-addressing
# Exercise 4 : it-network-plan-vlan
# Exercise 5 : it-network-plan-ipv4-static-routing


# Users 1,3,5,7,9,11 no recommendation
# Users 2,4,6,8,10,12 recommendation
def extract_entries(df: pd.DataFrame, was_recommended: bool):
    if was_recommended:
        eq = df.where(df.User % 2 != 0)
    else:
        eq = df.where(df.User % 2 == 0)

    return eq[eq["User"].notna()]


def prepare_data():
    data = pd.read_csv("preprocessed/main_preprocessed.csv")

    recommended = extract_entries(df=data, was_recommended=True)
    unrecommended = extract_entries(df=data, was_recommended=False)

    return recommended, unrecommended


def test_comparison_graphs(recommended, unrecommended):
    recommended_pretest = (
        recommended.groupby(["User", "ExerciseSkill"])
        .mean()["PretestCorrectRel"]
        .to_numpy()
    )
    recommended_posttest = (
        recommended.groupby(["User", "ExerciseSkill"])
        .mean()["PosttestCorrectRel"]
        .to_numpy()
    )
    recommended_pre_std = (
        recommended.groupby(["User", "ExerciseSkill"])
        .mean()
        .groupby(["User"])
        .std()["PretestCorrectRel"]
        .to_numpy()
    )
    recommended_post_std = (
        recommended.groupby(["User", "ExerciseSkill"])
        .mean()
        .groupby(["User"])
        .std()["PosttestCorrectRel"]
        .to_numpy()
    )

    unrecommended_pretest = (
        unrecommended.groupby(["User", "ExerciseSkill"])
        .mean()["PretestCorrectRel"]
        .to_numpy()
    )
    unrecommended_posttest = (
        unrecommended.groupby(["User", "ExerciseSkill"])
        .mean()["PosttestCorrectRel"]
        .to_numpy()
    )
    unrecommended_pre_std = (
        unrecommended.groupby(["User", "ExerciseSkill"])
        .mean()
        .groupby(["User"])
        .std()["PretestCorrectRel"]
        .to_numpy()
    )
    unrecommended_post_std = (
        unrecommended.groupby(["User", "ExerciseSkill"])
        .mean()
        .groupby(["User"])
        .std()["PosttestCorrectRel"]
        .to_numpy()
    )

    # Graphical Evaluation
    render_comparison_histogram(
        a=recommended_pretest,
        b=unrecommended_pretest,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Score",
        filename="comparison_pre_scores",
        output_dir="main/histograms/",
    )
    render_comparison_histogram(
        a=recommended_posttest,
        b=unrecommended_posttest,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Score",
        filename="comparison_post_scores",
        output_dir="main/histograms/",
    )
    render_comparison_histogram(
        a=recommended_pre_std,
        b=unrecommended_pre_std,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Standard Deviation",
        filename="comparison_pre_scores_std",
        output_dir="main/histograms/",
    )
    render_comparison_histogram(
        a=recommended_post_std,
        b=unrecommended_post_std,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Standard Deviation",
        filename="comparison_post_scores_std",
        output_dir="main/histograms",
    )


def test_improvement_normalized_change_skills(
    recommended, unrecommended, is_graph_norm, norm_val=0.05
):
    """
    Function to calculate if, and how significant the learning improvement for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    recommended_improvements = (
        recommended.groupby(["User", "ExerciseSkill"])
        .mean()["NormalizedChange"]
        .to_numpy()
    )
    unrecommended_improvements = (
        unrecommended.groupby(["User", "ExerciseSkill"])
        .mean()["NormalizedChange"]
        .to_numpy()
    )

    statistic, pvalue, test_name = perform_test(
        is_related=False,
        a=recommended_improvements,
        b=unrecommended_improvements,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Score",
        filename="improvement_normalized_change_skills",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative="greater",
        output_dir="main/histograms/",
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(recommended_improvements, unrecommended_improvements),
    )


def test_improvement_normalized_change_users(
    recommended, unrecommended, is_graph_norm, norm_val=0.05
):
    """
    Function to calculate if, and how significant the learning improvement for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_users.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    recommended_improvements = (
        recommended[["User", "NormalizedChange"]]
        .groupby(["User"])
        .mean()["NormalizedChange"]
        .to_numpy()
    )
    unrecommended_improvements = (
        unrecommended[["User", "NormalizedChange"]]
        .groupby(["User"])
        .mean()["NormalizedChange"]
        .to_numpy()
    )

    statistic, pvalue, test_name = perform_test(
        is_related=False,
        a=recommended_improvements,
        b=unrecommended_improvements,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Score",
        filename="improvement_normalized_change_users",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative="greater",
        output_dir="main/histograms/",
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(recommended_improvements, unrecommended_improvements),
    )


def test_reduced_recommendation_deviation_difference(
    recommended, unrecommended, is_graph_norm, norm_val=0.05
):
    """
    Function to calculate if, and how significant the reduction of deviation within the skills for users for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the user_deviation_difference.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """

    ## Calculate the mean scores for the pre and post test for each user and skill
    ## Flatten the data to have a single value for each user and skill
    recommended_vals = (
        recommended[["User", "ExerciseSkill", "PosttestCorrectRel"]]
        .groupby(["User", "ExerciseSkill"])
        .mean()
        .reset_index()
    )

    unrecommended_vals = (
        unrecommended[["User", "ExerciseSkill", "PosttestCorrectRel"]]
        .groupby(["User", "ExerciseSkill"])
        .mean()
        .reset_index()
    )

    ## Calculate the Coefficient of Variation (CV) for each user
    recommended_mean = (
        recommended_vals[["User", "PosttestCorrectRel"]]
        .groupby(["User"])
        .mean()["PosttestCorrectRel"]
    )
    recommended_std = (
        recommended_vals[["User", "PosttestCorrectRel"]]
        .groupby(["User"])
        .std()["PosttestCorrectRel"]
    )
    recommended_cv = (recommended_std / recommended_mean) * 100

    unrecommended_mean = (
        unrecommended_vals[["User", "PosttestCorrectRel"]]
        .groupby(["User"])
        .mean()["PosttestCorrectRel"]
    )
    unrecommended_std = (
        unrecommended_vals[["User", "PosttestCorrectRel"]]
        .groupby(["User"])
        .std()["PosttestCorrectRel"]
    )
    unrecommended_cv = (unrecommended_std / unrecommended_mean) * 100

    statistic, pvalue, test_name = perform_test(
        is_related=False,
        a=recommended_cv,
        b=unrecommended_cv,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Coefficient of Variation (CV)",
        filename="user_deviation_difference",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The CV is lower for the recommendation group indicating a reduction in deviation for the skill proficiency of the users
        alternative="less",
        output_dir="main/histograms/",
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(recommended_cv, unrecommended_cv),
    )


recommended, unrecommended = prepare_data()
test_comparison_graphs(recommended=recommended, unrecommended=unrecommended)

(
    normalized_skills_statistic,
    normalized_skills_pvalue,
    normalized_skills_test_name,
    normalized_skills_cohens,
) = test_improvement_normalized_change_skills(
    recommended, unrecommended, is_graph_norm=False, norm_val=0.05
)
(
    normalized_users_skills_statistic,
    normalized_users_skills_pvalue,
    normalized_users_skills_test_name,
    normalized_users_cohens,
) = test_improvement_normalized_change_users(
    recommended, unrecommended, is_graph_norm=False, norm_val=0.05
)
reduced_statistic, reduced_pvalue, reduced_test_name, reduced_cohens = (
    test_reduced_recommendation_deviation_difference(
        recommended=recommended,
        unrecommended=unrecommended,
        is_graph_norm=False,
        norm_val=0.05,
    )
)

data = pd.DataFrame(
    {
        "type": [
            "normalized_change_skills",
            "normalized_change_user",
            "reduced_deviation",
        ],
        "t": [
            normalized_skills_statistic,
            normalized_users_skills_statistic,
            reduced_statistic,
        ],
        "p": [
            normalized_skills_pvalue,
            normalized_users_skills_pvalue,
            reduced_pvalue,
        ],
        "cohens": [normalized_skills_cohens, normalized_users_cohens, reduced_cohens],
        "test": [
            normalized_skills_test_name,
            normalized_users_skills_test_name,
            reduced_test_name,
        ],
    }
)

data.to_csv(f"results/main_evaluation.csv", index=None)

render_boxplot(
    recommended.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    unrecommended.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    "main_boxplot_normalized_change",
    ["Recommended", "Unrecommended"],
    title="Normalized Change",
)
