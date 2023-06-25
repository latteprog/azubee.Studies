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

    for student in data["User"].unique():
        # Assuming student is defined
        df = data.loc[data["User"] == student][["ExerciseSkill","PretestCorrectRel","PosttestCorrectRel"]].groupby("ExerciseSkill").mean()
        df.reset_index(inplace=True)

        plot_pre_post(df=df, filename=f"pre/barplots/scores_{int(student)}", title=f'Scores for User: {int(student)}')

    trained = extract_entries(df=data, was_trained=True)
    not_trained = extract_entries(df=data, was_trained=False)

    return trained, not_trained

def test_improvement_abs_skill(trained, untrained, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the trained **skills** was, relative to the untrained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_abs_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_improvements = trained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()
    untrained_improvements = untrained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()

    t_test_result = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score",
        filename=f"pre/histograms/improvement_abs_skills",
        is_graph_norm=is_graph_norm, norm_val=norm_val
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_abs_exercise(trained, untrained, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the trained **exercises** was, relative to the untrained exercises.
    The is_graph_norm is an indicator, if both distributions within the improvement_abs_exercises.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_improvements = trained["ImprovementAbsNormalizedScores"].to_numpy()
    untrained_improvements = untrained["ImprovementAbsNormalizedScores"].to_numpy()

    t_test_result = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score",
        filename=f"pre/histograms/improvement_abs_exercises",
        is_graph_norm=is_graph_norm, norm_val=norm_val
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_normalized_change_skill(trained, untrained, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the trained **skills** in terms of normalized change was, relative to the untrained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_grouped = trained.groupby(["User", "ExerciseSkill"])
    untrained_grouped = untrained.groupby(["User", "ExerciseSkill"])

    trained_improvements = trained_grouped.mean()["NormalizedChange"].to_numpy()
    untrained_improvements = untrained_grouped.mean()["NormalizedChange"].to_numpy()

    t_test_result = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score",
        filename=f"pre/histograms/improvement_normalized_change_skills",
        is_graph_norm=is_graph_norm, norm_val=norm_val
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_normalized_change_exercise(trained, untrained, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the trained **exercises** in terms of normalized change was, relative to the untrained exercises.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_exercises.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    trained_improvements = trained["NormalizedChange"].to_numpy()
    untrained_improvements = untrained["NormalizedChange"].to_numpy()

    t_test_result = perform_test(
        is_related=True,
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score",
        filename=f"pre/histograms/improvement_normalized_change_exercises",
        is_graph_norm=is_graph_norm, norm_val=norm_val
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

trained, untrained = prepare_data()

improv_t_res, improv_cohens = test_improvement_abs_skill(trained, untrained, is_graph_norm=False, norm_val=0.05)
improv_exercise_t_res, improv_exercise_cohens = test_improvement_abs_exercise(trained, untrained, is_graph_norm=True, norm_val=0.05)
normalized_t_res, normalized_cohens = test_improvement_normalized_change_skill(trained, untrained, is_graph_norm=True, norm_val=0.05)
normalized_exercise_t_res, normalized_exercise_cohens = test_improvement_normalized_change_exercise(trained, untrained, is_graph_norm=False, norm_val=0.05)

data = pd.DataFrame({
    'type' : ["improvement_abs_skill", "improvement_abs_exercise", "normalized_change_skill", "normalized_change_exercise"], 
    't' : [improv_t_res.statistic, improv_exercise_t_res.statistic, normalized_t_res.statistic, normalized_exercise_t_res.statistic],
    'p': [improv_t_res.pvalue, improv_exercise_t_res.pvalue, normalized_t_res.pvalue, normalized_exercise_t_res.pvalue],
    'cohens': [improv_cohens, improv_exercise_cohens, normalized_cohens, normalized_exercise_cohens]
})
data.to_csv(f"results/pre_evaluation.csv", index=None)

render_boxplot(
    trained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    untrained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"],
    "pre/boxplot_normalized_change",
    ["Trained Skills", "Untrained Skills"],
    title="Normalized Change"
)