import pandas as pd
from scipy.stats import ttest_rel
from util import calculate_cohends_d, render_boxplot

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

def first_t_test(trained, untrained):
    trained_improvements = trained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbs"].to_numpy()
    untrained_improvements = untrained.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbs"].to_numpy()

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def second_t_test(trained, untrained):
    trained_improvements = trained["ImprovementAbs"].to_numpy()
    untrained_improvements = untrained["ImprovementAbs"].to_numpy()

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def third_t_test(trained, untrained):
    trained_improvements = trained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()
    untrained_improvements = untrained.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)

def fourth_t_test(trained, untrained):
    trained_improvements = trained["NormalizedChange"].to_numpy()
    untrained_improvements = untrained["NormalizedChange"].to_numpy()

    # Null hypothesis: trained and untrained improvements are equally distributed
    t_test_result = ttest_rel(
        trained_improvements,
        untrained_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(trained_improvements, untrained_improvements)


trained, untrained = prepare_data()

improv_t_res, improv_cohens = first_t_test(trained, untrained)
improv_exercise_t_res, improv_exercise_cohens = second_t_test(trained, untrained)
normalized_t_res, normalized_cohens = third_t_test(trained, untrained)
normalized_exercise_t_res, normalized_exercise_cohens = fourth_t_test(trained, untrained)


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
    "prestudy_normalized_change_boxplot",
    ["Trained Skills", "Untrained Skills"],
    title="Normalized Change"
)