import pandas as pd

def build_task_mapping(exercises: pd.DataFrame):
    mapping = dict()

    for _, row in exercises.iterrows():
        test = str(row["Test"])
        exercise = str(row["Exercise"])
        total = int(row["Total"])

        mapping.setdefault(test, dict())
        mapping[test][exercise] = total

    return mapping

def extract_data(study_name: str):
    xlsx = pd.ExcelFile(f"uni_augsburg_001/{study_name}/evaluation.xlsx")
    pretest = pd.read_excel(xlsx, "pretest")
    posttest = pd.read_excel(xlsx, "posttest")
    exercises = pd.read_excel(xlsx, "exercises")
    mapping = build_task_mapping(exercises)
    return pretest, posttest, mapping

def preprocess_evaluation(study_name: str, mapping_key: str):
    study_name = "pre_study"
    pretest, posttest, mapping = extract_data(study_name)

    data = pd.DataFrame()
    data["User"] = posttest["User"]
    data["Exercise"] = posttest["Exercise"]
    # Calculate improvement for each entry
    data["Improvement"] = (posttest["Correct"] - pretest["Correct"])

    for index, row in data.iterrows():
        exercise = str(row["Exercise"])
        # Divide by max points for relative improvement
        data.at[index, "Improvement"] = row["Improvement"] / mapping[mapping_key][exercise]

    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)


def preprocess_evaluation_main_study():
    study_name = "main_study"
    pretest, posttest, mapping = extract_data(study_name)

    data = pd.DataFrame()
    data["User"] = posttest["User"]
    data["Exercise"] = posttest["Exercise"]
    # Calculate improvement for each entry
    data["Improvement"] = (posttest["Correct"] - pretest["Correct"])

    for index, row in data.iterrows():
        exercise = str(row["Exercise"])
        # Divide by max points for relative improvement
        data.at[index, "Improvement"] = row["Improvement"] / mapping['2'][exercise]

    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)

preprocess_evaluation("pre_study", "1")
preprocess_evaluation("main_study", "2")