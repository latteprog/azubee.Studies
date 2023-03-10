import pandas as pd
from scipy.stats import ttest_ind
from util import render_boxplot, render_barplot


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


recommended, unrecommended = prepare_data()

render_skill_distributions(recommended, unrecommended)

# Null hypothesis: recommended and unrecommended improvements are equally distributed
t_test_result = ttest_ind(
    recommended["Improvement"].to_numpy(),
    unrecommended["Improvement"].to_numpy(),
    # Alternative: mean of the first is smaller than mean of the second distribution
    alternative='greater'
)

print(t_test_result)
render_boxplot(
    recommended,
    unrecommended,
    "mainstudy_improvements_boxplot",
    ["Recommendation System", "No recommendations"]
)
# TtestResult(statistic=1.5381972476555226, pvalue=0.06742065093784343, df=29)
# Konfidenzintervall 90 %
# p-Wert 0,0674

# Nullhypothese wird abgelehnt, Alternativhypothese wird angenommen
# Die recommended improvements sind im Schnitt größer als die unrecommended improvements
