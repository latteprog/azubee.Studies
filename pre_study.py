import pandas as pd
from scipy.stats import ttest_rel
from util import calculate_cohends_d, render_boxplot, render_comparison_histogram

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

def test_improvement_abs_skill(trained, untrained):
    trained_improvements = trained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()
    untrained_improvements = untrained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score", filename=f"pre_histogram_improvement_abs_skill")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_abs_exercise(trained, untrained):
    trained_improvements = trained["ImprovementAbsNormalizedScores"].to_numpy()
    untrained_improvements = untrained["ImprovementAbsNormalizedScores"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score", filename=f"pre_histogram_improvement_abs_exercise")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_normalized_change_skill(trained, untrained):
    trained_improvements = trained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()
    untrained_improvements = untrained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score", filename=f"pre_histogram_improvement_normalized_change_skill")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def test_improvement_normalized_change_exercise(trained, untrained):
    trained_improvements = trained["NormalizedChange"].to_numpy()
    untrained_improvements = untrained["NormalizedChange"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=trained_improvements,
        b=untrained_improvements,
        a_name="Trained",b_name="Untrained", x_label="Score", filename=f"pre_histogram_improvement_normalized_change_exercise")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)


trained, untrained = prepare_data()

improv_t_res, improv_cohens = test_improvement_abs_skill(trained, untrained)
improv_exercise_t_res, improv_exercise_cohens = test_improvement_abs_exercise(trained, untrained)
normalized_t_res, normalized_cohens = test_improvement_normalized_change_skill(trained, untrained)
normalized_exercise_t_res, normalized_exercise_cohens = test_improvement_normalized_change_exercise(trained, untrained)


data = pd.DataFrame({
    'type' : ["improvement_abs_skill", "improvement_abs_exercise", "normalized_change_skill", "normalized_change_exercise"], 
    't' : [improv_t_res.statistic, improv_exercise_t_res.statistic, normalized_t_res.statistic, normalized_exercise_t_res.statistic],
    'p': [improv_t_res.pvalue, improv_exercise_t_res.pvalue, normalized_t_res.pvalue, normalized_exercise_t_res.pvalue],
    'cohens': [improv_cohens, improv_exercise_cohens, normalized_cohens, normalized_exercise_cohens]
})
data.to_csv(f"results/pre_evaluation.csv", index=None)

render_boxplot(
    trained["NormalizedChange"],
    untrained["NormalizedChange"],
    "pre_boxplot_normalized_change",
    ["Trained Skills", "Untrained Skills"],
    title="Normalized Change"
)