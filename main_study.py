import pandas as pd
from scipy.stats import ttest_ind
from util import render_boxplot, render_barplot, calculate_cohends_d

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


def render_skill_distributions(recommended, unrecommended):
    recommended_means = recommended.groupby(["Exercise"]).mean()
    unrecommended_means = unrecommended.groupby(["Exercise"]).mean()

    recommended_correct = recommended_means["PosttestCorrectRel"].to_numpy()
    recommended_normalized_change = recommended_means["NormalizedChange"].to_numpy()
    unrecommended_correct = unrecommended_means["PosttestCorrectRel"].to_numpy()
    unrecommended_normalized_change = unrecommended_means["NormalizedChange"].to_numpy()

    # Dirty way of grouping skills [(1, 4), (2, 5), (3)]
    recommended_correct_grouped = [(recommended_correct[0] + recommended_correct[3]) / 2, (recommended_correct[1] + recommended_correct[4]), recommended_correct[2]]
    recommended_normalized_change_grouped = [(recommended_normalized_change[0] + recommended_normalized_change[3]) / 2, (recommended_normalized_change[1] + recommended_normalized_change[4]), recommended_normalized_change[2]]
    unrecommended_correct_grouped = [(unrecommended_correct[0] + unrecommended_correct[3]) / 2, (unrecommended_correct[1] + unrecommended_correct[4]), unrecommended_correct[2]]
    unrecommended_normalized_change_grouped = [(unrecommended_normalized_change[0] + unrecommended_normalized_change[3]) / 2, (unrecommended_normalized_change[1] + unrecommended_normalized_change[4]), unrecommended_normalized_change[2]]

    x = ["VLAN", "Static Routing", "IPv4 Addresses"]
    render_barplot(x, recommended_correct_grouped, "recommended_correct", "Recommendation System: Correct",)
    render_barplot(x, unrecommended_correct_grouped, "unrecommended_correct", "No Recommendation System: Correct")
    render_barplot(x, recommended_normalized_change_grouped, "recommended_normalized_change", "Recommendation System: Normalized Change")
    render_barplot(x, unrecommended_normalized_change_grouped, "unrecommended_normalized_change", "No Recommendation System: Normalized Change")


def first_t_test(recommended, unrecommended):
    recommended_improvements = recommended["ImprovementAbs"].to_numpy()
    unrecommended_improvements = unrecommended["ImprovementAbs"].to_numpy()

    # Null hypothesis: recommended and unrecommended improvements are equally distributed
    t_test_result = ttest_ind(
        recommended_improvements,
        unrecommended_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_improvements, unrecommended_improvements)

def second_t_test(recommended, unrecommended):
    recommended_normalized_change = recommended["NormalizedChange"].to_numpy()
    unrecommended_normalized_change = unrecommended["NormalizedChange"].to_numpy()

    # Null hypothesis: recommended and unrecommended variances in posttest correctness are equally distributed
    t_test_result = ttest_ind(
        recommended_normalized_change,
        unrecommended_normalized_change,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_normalized_change, unrecommended_normalized_change)

def third_t_test(recommended, unrecommended):
    # Group By Exercise
    recommended_by_exercise = recommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"]
    unrecommended_by_exercise = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"]

    # Minimum Skills
    recommended_min = recommended_by_exercise.groupby("User").agg('min')
    unrecommended_min = unrecommended_by_exercise.groupby("User").agg('min')

    # Maximum Skills
    recommended_max = recommended_by_exercise.groupby("User").agg('max')
    unrecommended_max = unrecommended_by_exercise.groupby("User").agg('max')

    # Differences
    recommended_diff = (recommended_max - recommended_min).to_numpy()
    unrecommended_diff = (unrecommended_max - unrecommended_min).to_numpy()
    
    # Null hypothesis: recommended and unrecommended diffs between max and min correctness are equally distributed
    t_test_result = ttest_ind(
        recommended_diff,
        unrecommended_diff,
        # Alternative: mean of the first is smaller than mean of the second distribution
        alternative='less'
    )

    return t_test_result, calculate_cohends_d(recommended_diff, unrecommended_diff)


recommended, unrecommended = prepare_data()
render_skill_distributions(recommended, unrecommended)

improv_t_res, improv_cohens = first_t_test(recommended, unrecommended)
normalized_t_res, normalized_cohens = second_t_test(recommended, unrecommended)
difference_t_res, difference_cohens = third_t_test(recommended, unrecommended)

data = pd.DataFrame({
    'type' : ["improvement_abs", "normalized_change", "difference"], 
    't' : [improv_t_res.statistic, normalized_t_res.statistic, difference_t_res.statistic],
    'p': [improv_t_res.pvalue, normalized_t_res.pvalue, difference_t_res.pvalue],
    'cohens': [improv_cohens, normalized_cohens, difference_cohens]
})
data.to_csv(f"results/main_evaluation.csv", index=None)


render_boxplot(
    recommended["NormalizedChange"].to_numpy(),
    unrecommended["NormalizedChange"].to_numpy(),
    "mainstudy_normalized_change_boxplot",
    ["Recommendation System", "No recommendations"],
    title="Normalized Change"
)
