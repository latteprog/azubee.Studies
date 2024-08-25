import pandas as pd
from util import calculate_cohends_d, render_boxplot, perform_test, plot_pre_post

# Exercise 1 : it-network-plan-vlan
# Exercise 2 : it-network-plan-ipv4-static-routing
# Exercise 3 : it-network-plan-vlan
# Exercise 4 : it-network-plan-ipv4-static-routing


# Users 1,3,5,7,9,11,13 it-network-plan-ipv4-static-routing
# Users 2,4,6,8,10,12 it-network-plan-vlan
def extract_entries(df: pd.DataFrame, was_trained: bool):
    if was_trained:
        eq = df.where(df.User % 2 != df.Exercise % 2)
    else:
        eq = df.where(df.User % 2 == df.Exercise % 2)

    return eq[eq["User"].notna()]


def prepare_data():
    data = pd.read_csv("preprocessed/pre_preprocessed.csv")

    trained = extract_entries(df=data, was_trained=True)
    not_trained = extract_entries(df=data, was_trained=False)

    return trained, not_trained


def test_improvement_normalized_change_skill(
    trained, untrained, is_graph_norm, norm_val=0.05
):
    """
    Function to calculate if, and how significant the learning improvement for the trained **skills** in terms of normalized change was, relative to the untrained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_grouped = trained.groupby(["User", "ExerciseSkill"])
    untrained_grouped = untrained.groupby(["User", "ExerciseSkill"])

    trained_improvements = trained_grouped.mean()["NormalizedChange"].to_numpy()
    untrained_improvements = untrained_grouped.mean()["NormalizedChange"].to_numpy()

    statistic, pvalue, test_name = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",
        b_name="Untrained",
        x_label="Score",
        filename="improvement_normalized_change_skills",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        output_dir="pre/histograms/",
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(trained_improvements, untrained_improvements),
    )


def test_improvement_normalized_change_exercise(
    trained, untrained, is_graph_norm, norm_val=0.05
):
    """
    Function to calculate if, and how significant the learning improvement for the trained **exercises** in terms of normalized change was, relative to the untrained exercises.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_exercises.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_improvements = trained["NormalizedChange"].to_numpy()
    untrained_improvements = untrained["NormalizedChange"].to_numpy()

    statistic, pvalue, test_name = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",
        b_name="Untrained",
        x_label="Score",
        filename=f"improvement_normalized_change_exercises",
        is_graph_norm=is_graph_norm,
        norm_val=norm_val,
        output_dir="pre/histograms/"
    )

    return (
        statistic,
        pvalue,
        test_name,
        calculate_cohends_d(trained_improvements, untrained_improvements),
    )


trained, untrained = prepare_data()

normalized_statistic, normalized_pvalue, normalized_test_name, normalized_cohens = (
    test_improvement_normalized_change_skill(
        trained, untrained, is_graph_norm=True, norm_val=0.05
    )
)
(
    normalized_exercise_statistic,
    normalized_exercise_pvalue,
    normalized_exercise_test_name,
    normalized_exercise_cohens,
) = test_improvement_normalized_change_exercise(
    trained, untrained, is_graph_norm=False, norm_val=0.05
)

data = pd.DataFrame(
    {
        "type": [
            "normalized_change_skill",
            "normalized_change_exercise",
        ],
        "t": [
            normalized_statistic,
            normalized_exercise_statistic,
        ],
        "p": [
            normalized_pvalue,
            normalized_exercise_pvalue,
        ],
        "cohens": [
            normalized_cohens,
            normalized_exercise_cohens,
        ],
        "test": [
            normalized_test_name,
            normalized_exercise_test_name,
        ],
    }
)
data.to_csv(f"results/pre_evaluation.csv", index=None)

render_boxplot(
    trained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    untrained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    "pre_boxplot_normalized_change",
    ["Trained (Faded) Skills", "Untrained (Unfaded) Skills"],
    title="Normalized Learning Gain",
)
