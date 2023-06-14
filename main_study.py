import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene, mannwhitneyu
from util import render_comparison_histogram, normalize_scores, calculate_cohends_d

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

def pre_test_correct_rel_test(recommended, unrecommended):
    recommended_pretest = recommended.groupby(["User", "ExerciseSkill"]).mean()["PretestCorrectRel"].to_numpy()
    recommended_posttest = recommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()

    unrecommended_pretest = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PretestCorrectRel"].to_numpy()
    unrecommended_posttest = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()

    # Graphical Evaluation
    render_comparison_histogram(a=recommended_pretest, b=unrecommended_pretest,a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main_histogram_comparison_pre_scores")
    render_comparison_histogram(a=recommended_posttest, b=unrecommended_posttest,a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main_histogram_comparison_post_scores")

def test_improvement_abs(recommended, unrecommended):
    recommended_improvements = recommended["ImprovementAbsNormalizedScores"].to_numpy()
    unrecommended_improvements = unrecommended["ImprovementAbsNormalizedScores"].to_numpy()

    # A) Graphical Evaluation
    render_comparison_histogram(
        a=recommended_improvements,
        b=unrecommended_improvements,
        a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main_histogram_improvement_abs")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test

    # Null hypothesis: recommended and unrecommended improvements are equally distributed
    t_test_result = ttest_ind(
        recommended_improvements,
        unrecommended_improvements,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_improvements, unrecommended_improvements)

def test_improvement_normalized_change(recommended, unrecommended):
    recommended_normalized_change = recommended["NormalizedChange"].to_numpy()
    unrecommended_normalized_change = unrecommended["NormalizedChange"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=recommended_normalized_change,
        b=unrecommended_normalized_change,
        a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main_histogram_improvement_normalized_change")

    # B) Analytical Evaluation (Distribution)

    # C) Hypothesis Test
    # Null hypothesis: recommended and unrecommended normalized change equally distributed (learning improvement)
    t_test_result = ttest_ind(
        recommended_normalized_change,
        unrecommended_normalized_change,
        # Alternative: mean of the first is greater than mean of the second distribution
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_normalized_change, unrecommended_normalized_change)

# This tests the overall group variance, not necessary insightful for "silowissen" for a single user
def test_smaller_recommendation_variances(recommended, unrecommended):
    # Group By Skill
    recommended_by_exercise = recommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()
    unrecommended_by_exercise = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=recommended_by_exercise,
        b=unrecommended_by_exercise,
        a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main_histogram_skill_rel")

    t_test_result = levene(recommended_by_exercise, unrecommended_by_exercise)

    variance_with_recommendation = np.var(recommended_by_exercise, ddof=1)
    variance_without_recommendation = np.var(unrecommended_by_exercise, ddof=1)

    print(variance_with_recommendation)
    print(variance_without_recommendation)

    if variance_with_recommendation < variance_without_recommendation:
        print("The recommendation system led to more even learning across the three skills.")
    else:
        print("The recommendation system did not lead to more even learning across the three skills.")

    return t_test_result

def test_reduced_recommendation_variances(recommended, unrecommended):
    recommended_variances = recommended.groupby(["User"]).var()["PosttestCorrectRel"].to_numpy()
    unrecommended_variances = unrecommended.groupby(["User"]).var()["PosttestCorrectRel"].to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=recommended_variances,
        b=unrecommended_variances,
        a_name="Recommended",b_name="Unrecommended", x_label="Variance", filename=f"main_histogram_user_variance")

    # C) Hypothesis Test
    # Null hypothesis: recommended and unrecommended changes in variance are equal
    t_test_result = mannwhitneyu(
        recommended_variances,
        unrecommended_variances,
        # Alternative: the distribution underlying the recommendation is stochastically less than the distribution underlying unrecommended
        alternative='less'
    )

    return t_test_result, calculate_cohends_d(recommended_variances, unrecommended_variances)

def test_reduced_recommendation_variances_difference(recommended, unrecommended):
    recommended_variances = recommended.groupby(["User"]).var()
    unrecommended_variances = unrecommended.groupby(["User"]).var()

    recommended_variances_differences = (recommended_variances["PosttestCorrectRel"] - recommended_variances["PretestCorrectRel"]).to_numpy()
    unrecommended_variances_differences = (unrecommended_variances["PosttestCorrectRel"] - unrecommended_variances["PretestCorrectRel"]).to_numpy()

    # A) Graphical Evaluation (Distribution)
    render_comparison_histogram(
        a=recommended_variances_differences,
        b=unrecommended_variances_differences,
        a_name="Recommended",b_name="Unrecommended", x_label="Variance", filename=f"main_histogram_user_variance_difference")

    # C) Hypothesis Test
    # Null hypothesis: recommended and unrecommended changes in variance are equal
    t_test_result = mannwhitneyu(
        recommended_variances_differences,
        unrecommended_variances_differences,
        # Alternative: the distribution underlying the recommendation is stochastically less than the distribution underlying unrecommended
        alternative='less'
    )

    return t_test_result, calculate_cohends_d(recommended_variances_differences, unrecommended_variances_differences)

recommended, unrecommended = prepare_data()
pre_test_correct_rel_test(recommended=recommended, unrecommended=unrecommended)

improv_t_res, improv_cohens = test_improvement_abs(recommended, unrecommended)
normalized_t_res, normalized_cohens = test_improvement_normalized_change(recommended, unrecommended)
smaller_t_res = test_smaller_recommendation_variances(recommended=recommended, unrecommended=unrecommended)
variance_t_res, variance_cohens = test_reduced_recommendation_variances(recommended=recommended, unrecommended=unrecommended)
reduced_t_res, reduced_cohens = test_reduced_recommendation_variances_difference(recommended=recommended, unrecommended=unrecommended)

data = pd.DataFrame({
    'type' : ["improvement_abs", "normalized_change", "smaller_variance", "reduced_variance", "variance"], 
    't' : [improv_t_res.statistic, normalized_t_res.statistic, smaller_t_res.statistic, reduced_t_res.statistic, variance_t_res.statistic],
    'p': [improv_t_res.pvalue, normalized_t_res.pvalue, smaller_t_res.pvalue, reduced_t_res.pvalue, variance_t_res.pvalue],
    'cohens': [improv_cohens, normalized_cohens,"",reduced_cohens, variance_cohens]
})

data.to_csv(f"results/main_evaluation.csv", index=None)