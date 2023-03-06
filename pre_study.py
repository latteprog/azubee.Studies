import pandas as pd
from scipy.stats import ttest_rel

def extract_entries(df: pd.DataFrame, was_trained: bool):
    if was_trained:
        eq = df.where(df.User % 2 != df.Exercise % 2)
    else:
        eq = df.where(df.User % 2 == df.Exercise % 2)

    return eq[eq["User"].notna()]

def prepare_data():
    data = pd.read_csv("preprocessed/pre_study_preprocessed.csv")

    trained = extract_entries(df=data, was_trained=True)["Improvement"].to_numpy()
    not_trained = extract_entries(df=data,was_trained=False)["Improvement"].to_numpy()

    return trained, not_trained


trained_improvements, not_trained_improvements = prepare_data()

# Null hypothesis: trained and untrained improvements are equally distributed
t_test_result = ttest_rel(
    trained_improvements,
    not_trained_improvements,
    # Alternative: mean of the first is greater than mean of the second distribution
    alternative='greater'
)

print(t_test_result)

# TtestResult(statistic=2.708711296686492, pvalue=0.006006691963389517, df=25)
# Konfidenzintervall 95 %
# p-Wert 0,006

# Nullhypothese wird abgelehnt, Alternativhypothese wird angenommen
# Die Trained Improvements sind im Schnitt größer als die Untrained Improvements