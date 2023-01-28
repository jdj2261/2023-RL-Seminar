import matplotlib.pyplot as plt


def plot_graph(rewards, xlabel="episode", ylabel="score"):
    plt.plot(rewards)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
