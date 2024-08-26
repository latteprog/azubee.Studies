import pandas as pd
import numpy as np
from data_preparation import normalized_change
from util import render_comparison_histogram, calculate_cohends_d, perform_test, plot_pre_post, render_boxplot
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
        df = data.loc[data["User"] == student][["ExerciseSkill","PretestCorrectRel","PosttestCorrectRel"]].groupby("ExerciseSkill").mean()
        df.reset_index(inplace=True)

        plot_pre_post(df=df, filename=f"main/barplots/scores_{int(student)}", title=f'Scores for User: {int(student)}')

    recommended = extract_entries(df=data, was_recommended=True)
    unrecommended = extract_entries(df=data, was_recommended=False)

    return recommended, unrecommended

def test_comparison_graphs(recommended, unrecommended):
    recommended_pretest = recommended.groupby(["User", "ExerciseSkill"]).mean()["PretestCorrectRel"].to_numpy()
    recommended_posttest = recommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()
    recommended_pre_std = recommended.groupby(["User", "ExerciseSkill"]).mean().groupby(["User"]).std()["PretestCorrectRel"].to_numpy()
    recommended_post_std = recommended.groupby(["User", "ExerciseSkill"]).mean().groupby(["User"]).std()["PosttestCorrectRel"].to_numpy()

    unrecommended_pretest = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PretestCorrectRel"].to_numpy()
    unrecommended_posttest = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["PosttestCorrectRel"].to_numpy()
    unrecommended_pre_std = unrecommended.groupby(["User", "ExerciseSkill"]).mean().groupby(["User"]).std()["PretestCorrectRel"].to_numpy()
    unrecommended_post_std = unrecommended.groupby(["User", "ExerciseSkill"]).mean().groupby(["User"]).std()["PosttestCorrectRel"].to_numpy()

    # Graphical Evaluation
    render_comparison_histogram(a=recommended_pretest, b=unrecommended_pretest,a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main/histograms/comparison_pre_scores")
    render_comparison_histogram(a=recommended_posttest, b=unrecommended_posttest,a_name="Recommended",b_name="Unrecommended", x_label="Score", filename=f"main/histograms/comparison_post_scores")
    render_comparison_histogram(a=recommended_pre_std, b=unrecommended_pre_std,a_name="Recommended",b_name="Unrecommended", x_label="Standard Deviation", filename=f"main/histograms/comparison_pre_scores_std")
    render_comparison_histogram(a=recommended_post_std, b=unrecommended_post_std,a_name="Recommended",b_name="Unrecommended", x_label="Standard Deviation", filename=f"main/histograms/comparison_post_scores_std")
        
def test_improvement_abs(recommended, unrecommended, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_abs_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    recommended_improvements = recommended.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()
    unrecommended_improvements = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["ImprovementAbsNormalizedScores"].to_numpy()

    t_test_result = perform_test(
        is_related=False,
        a=recommended_improvements,
        b=unrecommended_improvements,
        a_name="Recommended",b_name="Unrecommended", x_label="Score",
        filename=f"main/histograms/improvement_abs_skills",
        is_graph_norm=is_graph_norm, norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_improvements, unrecommended_improvements)

def test_improvement_normalized_change_skills(recommended, unrecommended, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_skills.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    recommended_improvements = recommended.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()
    unrecommended_improvements = unrecommended.groupby(["User", "ExerciseSkill"]).mean()["NormalizedChange"].to_numpy()

    t_test_result = perform_test(
        is_related=False,
        a=recommended_improvements,
        b=unrecommended_improvements,
        a_name="Recommended",b_name="Unrecommended", x_label="Score",
        filename=f"main/histograms/improvement_normalized_change_skills",
        is_graph_norm=is_graph_norm, norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative='greater'
    )
    
    return t_test_result, calculate_cohends_d(recommended_improvements, unrecommended_improvements)

# NEW FUNCTION
def test_improvement_normalized_change_users(recommended, unrecommended, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the learning improvement for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the improvement_normalized_change_users.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    full_score = 39

    # Sum scores for each user
    recommended_scores = recommended[["User","PretestCorrect", "PosttestCorrect"]].groupby(["User"]).sum()         
    unrecommended_scores = unrecommended[["User","PretestCorrect", "PosttestCorrect"]].groupby(["User"]).sum()

    # Calculate NLG from summed scores
    recommended_nlgs = [normalized_change(pre, post) for pre, post in zip(recommended_scores["PretestCorrect"].to_numpy() / full_score * 100, recommended_scores["PosttestCorrect"].to_numpy() / full_score * 100)]
    unrecommended_nlgs = [normalized_change(pre, post) for pre, post in zip(unrecommended_scores["PretestCorrect"].to_numpy() / full_score * 100, unrecommended_scores["PosttestCorrect"].to_numpy() / full_score * 100)]

    t_test_result = perform_test(
        is_related=False,
        a=recommended_nlgs,
        b=unrecommended_nlgs,
        a_name="Recommended",b_name="Unrecommended", x_label="Score",
        filename=f"main/histograms/improvement_normalized_change_users",
        is_graph_norm=is_graph_norm, norm_val=norm_val,
        # Alternative : The learning improvement of the recommended group is greater than the one of the unrecommended group
        alternative='greater'
    )

    render_boxplot(
        list(recommended_nlgs),
        list(unrecommended_nlgs),
        "main/boxplot_normalized_change",
        ["Recommendation", "Manual selection"],
        y_axis = "Normalized learning gain",
        title="Normalized Change"
    )

    print("NLG Rec", np.mean(recommended_nlgs), np.std(recommended_nlgs))
    print("NLG Sel", np.mean(unrecommended_nlgs), np.std(unrecommended_nlgs))
    
    return t_test_result, calculate_cohends_d(recommended_nlgs, unrecommended_nlgs)

def test_reduced_recommendation_deviation_difference(recommended, unrecommended, is_graph_norm, norm_val=0.05):
    """
    Function to calculate if, and how significant the reduction of deviation within the skills for users for the group using the recommendation system was, relative to the group manually selecting trained skills.
    The is_graph_norm is an indicator, if both distributions within the user_deviation_difference.png file are a normal distribution.
    This decides the statistical test used for evaluation.
    """
    recommended_vals = recommended[["User","PretestCorrectRel","PosttestCorrectRel"]].groupby(["User"]).std()
    unrecommended_vals = unrecommended[["User","PretestCorrectRel","PosttestCorrectRel"]].groupby(["User"]).std()
    
    recommended_differences = (recommended_vals["PretestCorrectRel"] - recommended_vals["PosttestCorrectRel"]).to_numpy()
    unrecommended_differences = (unrecommended_vals["PretestCorrectRel"] - unrecommended_vals["PosttestCorrectRel"]).to_numpy()

    t_test_result = perform_test(
        is_related=False,
        a=recommended_differences,
        b=unrecommended_differences,
        a_name="Recommended",b_name="Unrecommended", x_label="Standard Deviation",
        filename=f"main/histograms/user_deviation_difference",
        is_graph_norm=is_graph_norm, norm_val=norm_val,
        # Alternative : The reduction in standard deviation for users of the recommended group was greater than the for users of the unrecommended group
        alternative='greater'
    )

    return t_test_result, calculate_cohends_d(recommended_differences, unrecommended_differences)

# NEW FUNCTION
def test_reduced_skillgap(recommended, unrecommended, is_graph_norm, norm_val=0.05):
    # Scale summed score on each skill to [0, 1] range
    recommended_skill_scores = recommended[["User", "ExerciseSkill", "PosttestCorrect"]].groupby(["User", "ExerciseSkill"]).sum().reset_index()
    recommended_skill_scores.loc[recommended_skill_scores["ExerciseSkill"] == "it-network-plan-vlan", "PosttestCorrect"] /= 20
    recommended_skill_scores.loc[recommended_skill_scores["ExerciseSkill"] == "it-network-plan-ipv4-static-routing", "PosttestCorrect"] /= 13
    recommended_skill_scores.loc[recommended_skill_scores["ExerciseSkill"] == "it-network-plan-ipv4-addressing", "PosttestCorrect"] /= 6
    recommended_maxs = recommended_skill_scores[["User", "PosttestCorrect"]].groupby(["User"]).max()
    recommended_mins = recommended_skill_scores[["User", "PosttestCorrect"]].groupby(["User"]).min()

    unrecommended_skill_scores = unrecommended[["User", "ExerciseSkill", "PosttestCorrect"]].groupby(["User", "ExerciseSkill"]).sum().reset_index()
    unrecommended_skill_scores.loc[unrecommended_skill_scores["ExerciseSkill"] == "it-network-plan-vlan", "PosttestCorrect"] /= 20
    unrecommended_skill_scores.loc[unrecommended_skill_scores["ExerciseSkill"] == "it-network-plan-ipv4-static-routing", "PosttestCorrect"] /= 13
    unrecommended_skill_scores.loc[unrecommended_skill_scores["ExerciseSkill"] == "it-network-plan-ipv4-addressing", "PosttestCorrect"] /= 6
    unrecommended_maxs = unrecommended_skill_scores[["User", "PosttestCorrect"]].groupby(["User"]).max()
    unrecommended_mins = unrecommended_skill_scores[["User", "PosttestCorrect"]].groupby(["User"]).min()

    # Skill gap
    recommended_differences = (recommended_maxs["PosttestCorrect"] - recommended_mins["PosttestCorrect"]).to_numpy()
    unrecommended_differences = (unrecommended_maxs["PosttestCorrect"] - unrecommended_mins["PosttestCorrect"]).to_numpy()

    print("Rec skill gap", np.mean(recommended_differences), np.std(recommended_differences))
    print("Sel skill gap", np.mean(unrecommended_differences), np.std(unrecommended_differences))

    t_test_result = perform_test(
        is_related=False,
        a=recommended_differences,
        b=unrecommended_differences,
        a_name="Recommended",b_name="Unrecommended", x_label="Skill gap",
        filename=f"main/histograms/user_skill_gap",
        is_graph_norm=is_graph_norm, norm_val=norm_val,
        # Alternative : The reduction in standard deviation for users of the recommended group was greater than the for users of the unrecommended group
        alternative='less'
    )

    render_boxplot(
        list(recommended_differences),
        list(unrecommended_differences),
        "main/boxplot_skill_gap",
        ["Recommendation", "Manual selection"],
        y_axis = "Skill gap on post-test score",
        ylim = (-0.1, 1.1),
        title="Normalized Change"
    )

    return t_test_result, calculate_cohends_d(recommended_differences, unrecommended_differences)

recommended, unrecommended = prepare_data()
# test_comparison_graphs(recommended=recommended, unrecommended=unrecommended)

# improv_t_res, improv_cohens = test_improvement_abs(recommended, unrecommended, is_graph_norm=False, norm_val=0.05)
# normalized_skills_t_res, normalized_skills_cohens = test_improvement_normalized_change_skills(recommended, unrecommended, is_graph_norm=False, norm_val=0.05)
normalized_users_t_res, normalized_users_cohens = test_improvement_normalized_change_users(recommended, unrecommended, is_graph_norm=False, norm_val=0.05)
# reduced_t_res, reduced_cohens = test_reduced_recommendation_deviation_difference(recommended=recommended, unrecommended=unrecommended, is_graph_norm=False, norm_val=0.05)
gap_res, gap_cohens = test_reduced_skillgap(recommended=recommended, unrecommended=unrecommended, is_graph_norm=False, norm_val=0.05)

print("NLG", normalized_users_t_res, normalized_users_cohens)
print("Skill gap", gap_res, gap_cohens)

# data = pd.DataFrame({
#     'type' : ["improvement_abs", "normalized_change_skills", "normalized_change_user", "reduced_deviation", "skill_gap"], 
#     't' : [improv_t_res.statistic, normalized_skills_t_res.statistic, normalized_users_t_res.statistic, reduced_t_res.statistic, gap_res.statistic],
#     'p': [improv_t_res.pvalue, normalized_skills_t_res.pvalue, normalized_users_t_res.pvalue, reduced_t_res.pvalue, gap_res.pvalue],
#     'cohens': [improv_cohens, normalized_skills_cohens, normalized_users_cohens, reduced_cohens, gap_cohens]
# })

# data.to_csv(f"results/main_evaluation.csv", index=None)