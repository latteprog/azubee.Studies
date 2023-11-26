import pandas as pd
from util import render_comparison_histogram, render_comparison_histograms, normalize_scores

# Hake, R. R. (1998).
# https://doi.org/10.1119/1.18809
def normalized_gain(pretest, posttest):
    """
    Function to calculate the normalized gain
    The pretest and posttest scores are to be provided as percentages (e.g. 0,4 for 40%)
    """
    return (posttest - pretest) / (1 - pretest)

# Marx, J. D., & Cummings, K. (2007). Normalized change. American Journal of Physics, 75(1), 87–91.
# https://doi.org/10.1119/1.2372468
def normalized_change(pretest, posttest):
    """
    Function to calculate the normalized change
    The pretest and posttest scores are to be provided as percentages (e.g. 0,4 for 40%)
    """
    if posttest > pretest:
        return (posttest - pretest) / (100 - pretest)
    elif posttest < pretest:
        return (posttest - pretest) / (pretest)
    else:
        # "In the [perfect score] case we argue that this student’s scores should be removed from the data sets because the student’s performance is beyond the scope of the measurement instrument"
        # => We do not drop the student
        return 0

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
        pretest = raw_pretest.loc[raw_pretest["User"] != 10000]
        posttest = raw_posttest.loc[raw_posttest["User"] != 10000]
    elif study_name == "main":
        pretest = raw_pretest.loc[raw_pretest["User"] != 6]
        posttest = raw_posttest.loc[raw_posttest["User"] != 6]
    else:
        raise ValueError(study_name)

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

    for index, row in data.iterrows():
        exercise = int(row["Exercise"])
        data.at[index, "ExerciseSkill"] = skills[exercise - 1]

        # A) Relative Correctness of the exercise within the initial test
        pre_max_points = mapping[1][exercise]
        rel_pre_correct = row["PretestCorrect"] / pre_max_points * 100
        assert rel_pre_correct >= 0
        assert rel_pre_correct <= 100
        data.at[index, "PretestCorrectRel"] = rel_pre_correct

        # B) Relative Correctness of the exercise within the post test
        post_max_points = mapping[2][exercise]
        rel_post_correct = row["PosttestCorrect"] / post_max_points * 100
        assert rel_post_correct >= 0
        assert rel_post_correct <= 100
        data.at[index, "PosttestCorrectRel"] = rel_post_correct

        # C) Normalized Change 
        data.at[index, "NormalizedChange"] = normalized_change(pretest=rel_pre_correct, posttest=rel_post_correct)

    # D) Absolute Improvement by the difference of relative correctness within post and initial test
    data["ImprovementAbs"] = data["PosttestCorrectRel"] - data["PretestCorrectRel"]
    assert max(data["ImprovementAbs"]) <= 100
    assert min(data["ImprovementAbs"]) >= -100

    # E) Normalized Scores
    data["NormalizedPosttestCorrectRel"] = normalize_scores(data["PosttestCorrectRel"])
    data["NormalizedPretestCorrectRel"] = normalize_scores(data["PretestCorrectRel"])
    
    data["ImprovementAbsNormalizedScores"] = data["NormalizedPosttestCorrectRel"] - data["NormalizedPretestCorrectRel"]

    # Plot Some Histograms

    ## Relative Score Histogram
    render_comparison_histogram(
        a=data["PretestCorrectRel"],
        b=data["PosttestCorrectRel"],
        a_name="Pre",b_name="Post", x_label="Score", 
        filename=f"data/{study_name}/histogram_all_relative"
    )
    render_comparison_histogram(
        a=normalize_scores(data["PretestCorrectRel"]),
        b=normalize_scores(data["PosttestCorrectRel"]),
        a_name="Pre",b_name="Post", x_label="Score", 
        filename=f"data/{study_name}/histogram_all_relative_normalized"
    )

    ### Relative Score Histogram (VLAN)
    render_comparison_histogram(
        a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"],
        b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"],
        a_name="Pre (VLAN)",b_name="Post (VLAN)", x_label="Score", 
        filename=f"data/{study_name}/histogram_it-network-plan-vlan_relative"
    )
    
    ### Relative Score Histogram (Routing)
    render_comparison_histogram(
        a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"],
        b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"],
        a_name="Pre (Routing)",b_name="Post (Routing)", x_label="Score", 
        filename=f"data/{study_name}/histogram_it-network-plan-ipv4-static-routing_relative"
    )
    
    ### Relative Score Histogram (Addressing)
    if study_name == "main":
        render_comparison_histogram(
            a=data["PretestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-addressing"],
            b=data["PosttestCorrectRel"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-addressing"],
            a_name="Pre (Addressing)",b_name="Post (Addressing)", x_label="Score", 
            filename=f"data/{study_name}/histogram_it-network-plan-ipv4-addressing_relative"
        )
    
    ## Normalized Change Histogram
    if study_name == "main":
        render_comparison_histograms(
            data_list= [
                {"name": "VLAN", "values": data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"]},
                {"name": "IPv4 Routing", "values": data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"]},
                {"name": "IPv4 Addressing", "values": data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-addressing"]}
            ],
            x_label="Normalized Learning Gain",
            filename=f"data/{study_name}/histogram_normalized_change"
        )
    else:
        render_comparison_histograms(
            data_list= [
                {"name": "VLAN", "values": data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-vlan"]},
                {"name": "IPv4 Routing", "values": data["NormalizedChange"].loc[data["ExerciseSkill"] == "it-network-plan-ipv4-static-routing"]}
            ],
            x_label="Normalized Learning Gain",
            filename=f"data/{study_name}/histogram_normalized_change"
        )
        
    ## Absolute Improvement Histogram
    render_comparison_histogram(
        a=data["ImprovementAbs"],
        b=data["ImprovementAbsNormalizedScores"],
        a_name="Abs", b_name="Normalized", x_label="Improvement", 
        filename=f"data/{study_name}/histogram_improvement"
    )
    
    data.to_csv(f"preprocessed/{study_name}_preprocessed.csv", index=None)

# The pre test where the exercise skill ordering was :
# vlan, routing, vlan, routing
preprocess_evaluation("pre", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])
# The pre test where the exercise skill ordering was :
# vlan, routing, addressing, vlan, routing
preprocess_evaluation("main", ["it-network-plan-vlan","it-network-plan-ipv4-static-routing","it-network-plan-ipv4-addressing","it-network-plan-vlan","it-network-plan-ipv4-static-routing"])