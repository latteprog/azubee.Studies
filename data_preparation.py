import pandas as pd

def build_task_mapping(exercises: pd.DataFrame):
    mapping = dict()

    for _, row in exercises.iterrows():
        test = int(row["Test"])
        exercise = int(row["Exercise"])
        total = int(row["Total"])

        mapping.setdefault(test, dict())
        mapping[test][exercise] = total

    return mapping

def extract_data(study_name: str):
    xlsx = pd.ExcelFile(f"data/{study_name}_evaluation.xlsx")
    pretest = pd.read_excel(xlsx, "pretest")
    posttest = pd.read_excel(xlsx, "posttest")
    exercises = pd.read_excel(xlsx, "exercises")
    mapping = build_task_mapping(exercises)
    return pretest, posttest, mapping

def preprocess_evaluation(study_name, skills):
    pretest, posttest, mapping = extract_data(study_name)

    data = pd.DataFrame()
    
    data["User"] = posttest["User"]
    data["Exercise"] = posttest["Exercise"]

    data["PretestCorrect"] = pretest["Correct"]
    data["PosttestCorrect"] = posttest["Correct"]

    # A) Absolute Improvement of users from initial to post test
    data["ImprovementAbs"] = posttest["Correct"] - pretest["Correct"]

    for index, row in data.iterrows():
        exercise = int(row["Exercise"])
        data.at[index, "ExerciseSkill"] = skills[exercise - 1]

        # A) Relative Correctness of the exercise within the initial test
        pre_max_points = mapping[1][exercise]
        rel_pre_correct = row["PretestCorrect"] / pre_max_points
        assert rel_pre_correct >= 0
        assert rel_pre_correct <= 1
        data.at[index, "PretestCorrectRel"] = rel_pre_correct

        # B) Relative Correctness of the exercise within the post test
        post_max_points = mapping[2][exercise]
        rel_post_correct = row["PosttestCorrect"] / post_max_points
        assert rel_post_correct >= 0
        assert rel_post_correct <= 1
        data.at[index, "PosttestCorrectRel"] = rel_post_correct

        # C) Absolute Improvement by the difference of relative correctness within post and initial test
        improvement = rel_post_correct - rel_pre_correct
        assert improvement <= 1
        assert improvement >= -1
        data.at[index, "ImprovementAbs"] = improvement

        # D) Normalized Change (see https://www.physport.org/recommendations/Entry.cfm?ID=93334)
        if rel_post_correct > rel_pre_correct:
            data.at[index, "NormalizedChange"] = (rel_post_correct - rel_pre_correct) / (1 - rel_pre_correct)
        elif rel_post_correct < rel_pre_correct:
            data.at[index, "NormalizedChange"] = (rel_post_correct - rel_pre_correct) / (rel_pre_correct)
        else:
            data.at[index, "NormalizedChange"] = 0

    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)

preprocess_evaluation("pre", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])
preprocess_evaluation("main", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-ipv4-addressing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])