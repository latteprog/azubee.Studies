import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import Dict, List

SKILL_ID = "it-network-plan"

datapoints: Dict[str, List[float]] = dict()

for file in os.listdir("abzuege"):
    data: pd.DataFrame = pd.read_csv(os.path.join("abzuege", file), sep=";")
    mastery_values = data.where(data["SkillId"] == SKILL_ID).dropna()

    for _, row in mastery_values.iterrows():
        datapoints.setdefault(row["UserId"], list())
        datapoints[str(row["UserId"])].append(float(row["Mastery"]))

for index, key in enumerate(datapoints.keys()):
    name = f"User {index}"
    x = range(len(datapoints[key]))
    y = datapoints[key]

    plt.plot(x, y, label=name)

plt.show()
