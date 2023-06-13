import pandas as pd
import numpy as np
from util import render_comparison_histogram, normalize_scores

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

    # The user 1 had extraordinaly bad results in the pre study (practically all 0s)
    # The user 6 had extraordinaly bad results in the main study (all 0s)
    raw_pretest = pd.read_excel(xlsx, "pretest")
    raw_posttest = pd.read_excel(xlsx, "posttest")

    if study_name == "pre":
        pretest = raw_pretest.loc[raw_pretest["User"] != 1]
        posttest = raw_posttest.loc[raw_posttest["User"] != 1]
    elif study_name == "main":
        pretest = raw_pretest.loc[raw_pretest["User"] != 6]
        posttest = raw_posttest.loc[raw_posttest["User"] != 6]
    else:
        raise ValueError(study_name)

    exercises = pd.read_excel(xlsx, "exercises")
    mapping = build_task_mapping(exercises)
    return pretest, posttest, mapping

def preprocess_evaluation(study_name, skills):
    exercise_skill_scores = {}
    pretest, posttest, mapping = extract_data(study_name)
    data = pd.DataFrame()
    
    data["User"] = posttest["User"]
    data["Exercise"] = posttest["Exercise"]

    data["PretestCorrect"] = pretest["Correct"]
    data["PosttestCorrect"] = posttest["Correct"]

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

        # C) Normalized Change (see https://www.physport.org/recommendations/Entry.cfm?ID=93334)
        if rel_post_correct > rel_pre_correct:
            data.at[index, "NormalizedChange"] = (rel_post_correct - rel_pre_correct) / (1 - rel_pre_correct)
        elif rel_post_correct < rel_pre_correct:
            data.at[index, "NormalizedChange"] = (rel_post_correct - rel_pre_correct) / (rel_pre_correct)
        else:
            data.at[index, "NormalizedChange"] = 0

        ## Store Score
        if not data.at[index, "ExerciseSkill"] in exercise_skill_scores:
            exercise_skill_scores[data.at[index, "ExerciseSkill"]] = {
                "pre" : [],
                "post": []
            }

        exercise_skill_scores[data.at[index, "ExerciseSkill"]]["pre"].append(data.at[index, "PretestCorrectRel"])
        exercise_skill_scores[data.at[index, "ExerciseSkill"]]["post"].append(data.at[index, "PosttestCorrectRel"])
    
    # D) Absolute Improvement by the difference of relative correctness within post and initial test
    data["ImprovementAbs"] = data["PosttestCorrectRel"] - data["PretestCorrectRel"]
    assert max(data["ImprovementAbs"]) <= 1
    assert min(data["ImprovementAbs"]) >= -1

    # E) Normalized Scores
    data["NormalizedPosttestCorrectRel"] = normalize_scores(data["PosttestCorrectRel"])
    data["NormalizedPretestCorrectRel"] = normalize_scores(data["PretestCorrectRel"])
    
    data["ImprovementAbsNormalizedScores"] = data["NormalizedPosttestCorrectRel"] - data["NormalizedPretestCorrectRel"]

    # Plot Some Histograms

    ## Relative Score Histogram
    render_comparison_histogram(a=data["PretestCorrectRel"],b=data["PosttestCorrectRel"],a_name="Pre",b_name="Post", x_label="Score", filename=f"{study_name}_histogram_all_relative")
    render_comparison_histogram(a=normalize_scores(data["PretestCorrectRel"]),b=normalize_scores(data["PosttestCorrectRel"]),a_name="Pre",b_name="Post", x_label="Score", filename=f"{study_name}_histogram_all_relative_normalized")

    ### Relative Score Histogram (VLAN)
    render_comparison_histogram(
        a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"],
        b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"],
        a_name="Pre (VLAN)",b_name="Post (VLAN)", x_label="Score", filename=f"{study_name}_histogram_it-network-plan-vlan_relative")
    
    ### Relative Score Histogram (Routing)
    render_comparison_histogram(
        a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"],
        b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"],
        a_name="Pre (Routing)",b_name="Post (Routing)", x_label="Score", filename=f"{study_name}_histogram_it-network-plan-ipv4-static-routing_relative")
    
    ### Relative Score Histogram (Routing)
    if study_name == "main":
        render_comparison_histogram(
            a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-addressing"],
            b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-addressing"],
            a_name="Pre (Routing)",b_name="Post (Routing)", x_label="Score", filename=f"{study_name}_histogram_it-network-plan-ipv4-addressing_relative")
    
    ## Normalized Change Histogram
    render_comparison_histogram(
        a=data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"],
        b=data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"],
        a_name="VLAN",b_name="Routing", x_label="Normalized Change", filename=f"{study_name}_histogram_normalized_change")
    
    ## Absolute Improvement Histogram
    render_comparison_histogram(a=data["ImprovementAbs"],b=data["ImprovementAbsNormalizedScores"],a_name="Abs",b_name="Normalized", x_label="Improvement", filename=f"{study_name}_histogram_improvement")
    
    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)


# The pre test where the exercise skill ordering was :
# vlan, routing, vlan, routing
preprocess_evaluation("pre", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])
# The pre test where the exercise skill ordering was :
# vlan, routing, addressing, vlan, routing
preprocess_evaluation("main", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-ipv4-addressing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])