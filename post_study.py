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

    recommended_correct = recommended_means["PosttestCorrectRel"]
    recommended_improvement = recommended_means["Improvement"]
    unrecommended_correct = unrecommended_means["PosttestCorrectRel"]
    unrecommended_improvement = unrecommended_means["Improvement"]

    x = [1, 2, 3, 4, 5]
    render_barplot(x, recommended_correct.to_numpy(), "recommended_correct")
    render_barplot(x, unrecommended_correct.to_numpy(), "unrecommended_correct")
    render_barplot(x, recommended_improvement.to_numpy(), "recommended_improvement")
    render_barplot(x, unrecommended_improvement.to_numpy(), "unrecommended_improvement")


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
    recommended_correct_var = recommended.groupby(["User"]).var()["PosttestCorrectRel"]
    unrecommended_correct_var = unrecommended.groupby(["User"]).var()["PosttestCorrectRel"]

    # Null hypothesis: recommended and unrecommended variances in posttest correctness are equally distributed
    t_test_result = ttest_ind(
        recommended_correct_var,
        unrecommended_correct_var,
        # Alternative: mean of the first is smaller than mean of the second distribution
        alternative='less'
    )

    print("3. Test if students with recommendation system learned distributed more equally")
    print("   ", t_test_result)
    print(f"    Cohens D value: {calculate_cohends_d(recommended_correct_var, unrecommended_correct_var)}")


recommended, unrecommended = prepare_data()
render_skill_distributions(recommended, unrecommended)


first_t_test(recommended, unrecommended)
second_t_test(recommended, unrecommended)
third_t_test(recommended, unrecommended)


render_boxplot(
    recommended,
    unrecommended,
    "mainstudy_improvements_boxplot",
    ["Recommendation System", "No recommendations"]
)
