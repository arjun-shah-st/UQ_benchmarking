"""
plotting.py
===========
A collection of helper functions for plotting error metrics.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def load_performance_data(path, typ="frame"):
    with open(path, 'rb') as f:
        df = pd.read_json(f, orient='split', typ=typ)
    return df


def plot_uniform_adaptive(path_dict, uniform=True, adaptive=True, filename=None,
                          x_lab="n_samples", y_lab="mse", save_type=".png", **kwargs):

    if not (uniform or adaptive):
        raise ValueError("plot uniform_adaptive must have at least one of uniform or adaptive datasets to plot")
    if uniform:
        u_path = path_dict["uniform"]
        u_df = load_performance_data(u_path)
        sns.lineplot(x=x_lab, y=y_lab, data=u_df, label="uniform")
        save_path = os.path.dirname(os.path.dirname(u_path))

    if adaptive:
        a_path = path_dict["adaptive"]
        a_df = load_performance_data(a_path)
        sns.lineplot(x=x_lab, y=y_lab, data=a_df, label="adaptive")
        save_path = os.path.dirname(os.path.dirname(a_path))

    plt.semilogy()
    print (save_path)
    plt.xlabel('Number of Samples')
    plt.ylabel(y_lab)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, y_lab + save_type))
    plt.show()


def batch_plot(project_path, projects_list, uniform=True, adaptive=True, *args, **kwargs):

    for project in projects_list:
        path = os.path.join(project_path, project)

        path_dict = {}
        if uniform is True or adaptive is True:
            if uniform is True:
                path_dict["uniform"] = os.path.join(path, "Uniform", "performance_data.json")
            if adaptive is True:
                path_dict["adaptive"] = os.path.join(path, "Adaptive", "performance_data.json")
        else:
            path_dict["data"] = project

        plot_uniform_adaptive(path_dict, uniform, adaptive, *args, **kwargs)


if __name__ == "__main__":
    PROJECT_PATH = os.path.join("D:", os.sep, "UBAS", "projects")
    projects = ["2D_central_peak", "3D_central_peak", "4D_central_peak", "5D_central_peak"]
    batch_plot(PROJECT_PATH, projects, y_lab='mse')
