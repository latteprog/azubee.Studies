import pandas as pd
from scipy.stats import ttest_rel
from util import render_boxplot


def extract_entries(df: pd.DataFrame, was_trained: bool):
    if was_trained:
        eq = df.where(df.User % 2 != df.Exercise % 2)
    else:
        eq = df.where(df.User % 2 == df.Exercise % 2)

    return eq[eq["User"].notna()]


def prepare_data():
    data = pd.read_csv("preprocessed/pre_study_preprocessed.csv")

    trained = extract_entries(df=data, was_trained=True)["Improvement"].to_numpy()
    not_trained = extract_entries(df=data, was_trained=False)["Improvement"].to_numpy()

    return trained, not_trained


trained_improvements, untrained_improvements = prepare_data()

# Null hypothesis: trained and untrained improvements are equally distributed
t_test_result = ttest_rel(
    trained_improvements,
    untrained_improvements,
    # Alternative: mean of the first is greater than mean of the second distribution
    alternative='greater'
)

print(t_test_result)

render_boxplot(
    trained_improvements,
    untrained_improvements,
    "prestudy_improvements_boxplot",
    ["Trained", "Untrained"]
)
