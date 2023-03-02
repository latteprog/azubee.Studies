import pandas as pd
import numpy as np

def build_task_mapping(exercises: pd.DataFrame):
    mapping = dict()

    for _, row in exercises.iterrows():
        test = str(row["Test"])
        exercise = str(row["Exercise"])
        total = int(row["Total"])

        mapping.setdefault(test, dict())
        mapping[test][exercise] = total

    return mapping

def preprocess_evaluation(study_name: str, study_num: str):
    xlsx = pd.ExcelFile(f"uni_augsburg_001/{study_name}/evaluation.xlsx")
    pretest = pd.read_excel(xlsx, "pretest")
    posttest = pd.read_excel(xlsx, "posttest")
    exercises = pd.read_excel(xlsx, "exercises")
    mapping = build_task_mapping(exercises)

    data = pd.DataFrame()
    data["User"] = posttest["User"]
    data["Exercise"] = posttest["Exercise"]
    data["Improvement"] = (posttest["Correct"] - pretest["Correct"])

    for index, row in data.iterrows():
        exercise = str(row["Exercise"])
        data.at[index, "Improvement"] = row["Improvement"] / mapping[study_num][exercise]

    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)

preprocess_evaluation("pre_study", '1')
preprocess_evaluation("main_study", '2')