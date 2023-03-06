import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def render_boxplot(trained, untrained):
    ax = sns.boxplot(data=[trained, untrained], medianprops={'color': 'purple', 'lw': 2})
    ax.set_xticklabels(["Recommendations", "No recommendations"])
    plt.show()

def print_stats(trained: np.ndarray, untrained: np.ndarray):
    print("\n")
    print("Trained Improvements:")
    print("Max", max(trained))
    print("Min", min(trained))
    print("Mean", trained.mean())
    print("Median", np.median(trained))
    print("\n")
    print("Untrained Improvements:")
    print("Max", max(untrained))
    print("Min", min(untrained))
    print("Mean", untrained.mean())
    print("Median", np.median(untrained))