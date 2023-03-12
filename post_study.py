import pandas as pd
from scipy.stats import ttest_ind
from util import render_boxplot, render_barplot, calculate_cohends_d


def extract_entries(df: pd.DataFrame, was_recommended: bool):
    if was_recommended:
        eq = df.where(df.User % 2 != 0)
    else:
        eq = df.where(df.User % 2 == 0)

    return eq[eq["User"].notna()]


def prepare_data():
    data = pd.read_csv("preprocessed/main_study_preprocessed.csv")

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

    x = [1, 2, 3]
    labels = ["VLAN", "Static Routing", "IPv4 Addresses"]
    render_barplot(x, recommended_correct_grouped, "recommended_correct", "Recommendation System: Correct", labels=labels)
    render_barplot(x, unrecommended_correct_grouped, "unrecommended_correct", "No Recommendation System: Correct", labels=labels)
    render_barplot(x, recommended_normalized_change_grouped, "recommended_normalized_change", "Recommendation System: Normalized Change", labels=labels)
    render_barplot(x, unrecommended_normalized_change_grouped, "unrecommended_normalized_change", "No Recommendation System: Normalized Change", labels=labels)


def first_t_test(recommended, unrecommended):
    recommended_improvements = recommended["Improvement"].to_numpy()
    unrecommended_improvements = unrecommended["Improvement"].to_numpy()

    # Null hypothesis: recommended and unrecommended improvements are equally distributed
    t_test_result = ttest_ind(
        recommended_improvements,
        unrecommended_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    print("1. Test if students with recommendation system improved more")
    print("   ", t_test_result)
    print(f"    Cohens D value: {calculate_cohends_d(recommended_improvements, unrecommended_improvements)}")


def second_t_test(recommended, unrecommended):
    recommended_correct_var = recommended["NormalizedChange"].to_numpy()
    unrecommended_correct_var = unrecommended["NormalizedChange"].to_numpy()

    # Null hypothesis: recommended and unrecommended variances in posttest correctness are equally distributed
    t_test_result = ttest_ind(
        recommended_correct_var,
        unrecommended_correct_var,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    # https://www.physport.org/recommendations/Entry.cfm?ID=93334
    print("2. Test if students with recommendation had a higher normalized change (similar to average student normalized gain)")
    print("   ", t_test_result)
    print(f"    Cohens D value: {calculate_cohends_d(recommended_correct_var, unrecommended_correct_var)}")


def third_t_test(recommended, unrecommended):
    recommended_correct_max = recommended.groupby(["User"]).max()["PosttestCorrectRel"]
    unrecommended_correct_max = unrecommended.groupby(["User"]).max()["PosttestCorrectRel"]
    recommended_correct_min = recommended.groupby(["User"]).min()["PosttestCorrectRel"]
    unrecommended_correct_min = unrecommended.groupby(["User"]).min()["PosttestCorrectRel"]

    recommended_correct_diff = (recommended_correct_max - recommended_correct_min).to_numpy()
    unrecommended_correct_diff = (unrecommended_correct_max - unrecommended_correct_min).to_numpy()

    # Null hypothesis: recommended and unrecommended diffs between max and min correctness are equally distributed
    t_test_result = ttest_ind(
        recommended_correct_diff,
        unrecommended_correct_diff,
        # Alternative: mean of the first is smaller than mean of the second distribution
        alternative='less'
    )

    print("3. Test if students with recommendation system learned distributed more equally")
    print("   ", t_test_result)
    print(f"    Cohens D value: {calculate_cohends_d(recommended_correct_diff, unrecommended_correct_diff)}")


recommended, unrecommended = prepare_data()
render_skill_distributions(recommended, unrecommended)


first_t_test(recommended, unrecommended)
second_t_test(recommended, unrecommended)
third_t_test(recommended, unrecommended)


render_boxplot(
    recommended["NormalizedChange"].to_numpy(),
    unrecommended["NormalizedChange"].to_numpy(),
    "mainstudy_normalized_change_boxplot",
    ["Recommendation System", "No recommendations"],
    title="Normalized Change"
)
