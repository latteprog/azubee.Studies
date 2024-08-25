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

    for student in data["User"].unique():
        # Assuming student is defined
        df = (
            data.loc[data["User"] == student][
                ["ExerciseSkill", "PretestCorrectRel", "PosttestCorrectRel"]
            ]
            .groupby("ExerciseSkill")
            .mean()
        )
        df.reset_index(inplace=True)

        plot_pre_post(
            df=df,
            filename=f"main/barplots/scores_{int(student)}",
            title=f"Scores for User: {int(student)}",
        )

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
        filename=f"main/histograms/comparison_pre_scores",
    )
    render_comparison_histogram(
        a=recommended_posttest,
        b=unrecommended_posttest,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Score",
        filename=f"main/histograms/comparison_post_scores",
    )
    render_comparison_histogram(
        a=recommended_pre_std,
        b=unrecommended_pre_std,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Standard Deviation",
        filename=f"main/histograms/comparison_pre_scores_std",
    )
    render_comparison_histogram(
        a=recommended_post_std,
        b=unrecommended_post_std,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Standard Deviation",
        filename=f"main/histograms/comparison_post_scores_std",
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
        filename=f"main/histograms/improvement_normalized_change_skills",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative="greater",
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
        filename=f"main/histograms/improvement_normalized_change_users",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative="greater",
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
    recommended_vals = (
        recommended[["User", "PretestCorrectRel", "PosttestCorrectRel"]]
        .groupby(["User"])
        .std()
    )
    unrecommended_vals = (
        unrecommended[["User", "PretestCorrectRel", "PosttestCorrectRel"]]
        .groupby(["User"])
        .std()
    )

    recommended_differences = (
        recommended_vals["PretestCorrectRel"] - recommended_vals["PosttestCorrectRel"]
    ).to_numpy()
    unrecommended_differences = (
        unrecommended_vals["PretestCorrectRel"]
        - unrecommended_vals["PosttestCorrectRel"]
    ).to_numpy()

    statistic, pvalue, test_name = perform_test(
        is_related=False,
        a=recommended_differences,
        b=unrecommended_differences,
        a_name="Recommended",
        b_name="Unrecommended",
        x_label="Standard Deviation",
        filename=f"main/histograms/user_deviation_difference",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        # Alternative : The reduction in standard deviation for users of the recommended group was greater than the for users of the unrecommended group
        alternative="greater",
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(recommended_differences, unrecommended_differences),
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
    "main/boxplot_normalized_change",
    ["Recommended", "Unrecommended"],
    title="Normalized Change",
)
