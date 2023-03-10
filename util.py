import seaborn as sns
import matplotlib.pyplot as plt


def render_boxplot(trained, untrained, filename, labels):
    ax = sns.boxplot(data=[trained, untrained], medianprops={'color': 'purple', 'lw': 2})
    ax.set_xticklabels(labels)
    plt.savefig(f"img/{filename}.png")
    plt.close()


def render_barplot(x, y, filename):
    sns.barplot(x=x, y=y)
    plt.savefig(f"img/{filename}.png")
    plt.close()
