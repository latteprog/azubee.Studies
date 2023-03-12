import pandas as pd
from scipy.stats import ttest_rel
from util import render_boxplot, calculate_cohends_d


def extract_entries(df: pd.DataFrame, was_trained: bool):
    if was_trained:
        eq = df.where(df.User % 2 != df.Exercise % 2)
    else:
        eq = df.where(df.User % 2 == df.Exercise % 2)

    return eq[eq["User"].notna()]


def prepare_data():
    data = pd.read_csv("preprocessed/pre_study_preprocessed.csv")

    trained = extract_entries(df=data, was_trained=True)["NormalizedChange"].to_numpy()
    not_trained = extract_entries(df=data, was_trained=False)["NormalizedChange"].to_numpy()

    return trained, not_trained


trained, untrained = prepare_data()

# Null hypothesis: trained and untrained improvements are equally distributed
t_test_result = ttest_rel(
    trained,
    untrained,
    # Alternative: mean of the first is greater than mean of the second distribution
    alternative='greater'
)

print(t_test_result)
print(f"    Cohens D value: {calculate_cohends_d(trained, untrained)}")

render_boxplot(
    trained,
    untrained,
    "prestudy_normalized_change_boxplot",
    ["Trained", "Untrained"]
)
