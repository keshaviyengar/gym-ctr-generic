import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def process_files_and_get_dataframes(system_idx, experiments):

    files = []
    dfs = []
    for name in experiments:
        for id in system_idx:
            f = project_folder + name + "/evaluations_" + str(id) + ".csv"
            df = pd.read_csv(f)
            dg = np.array([df['desired_goal_x'], df['desired_goal_y'], df['desired_goal_z']])
            ag = np.array([df['achieved_goal_x'], df['achieved_goal_y'], df['achieved_goal_z']])
            sg = np.array([df['starting_position_x'], df['starting_position_y'], df['starting_position_z']])
            df['errors_pos'] = np.linalg.norm(np.transpose(ag - dg), axis=1) * 1000
            df['goal_dist'] = np.linalg.norm(np.transpose(sg - dg), axis=1) * 1000
            df['success'] = df['errors_pos'] < 1.0
            df["system"] = id
            df['name'] = name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_alpha_box_plots(df, alpha):
    sns.boxplot(x="alpha_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Alpha " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.ylim([0, 5])
    plt.show()

def plot_B_box_plots(df, alpha):
    sns.boxplot(x="B_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Beta " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.show()


if __name__ == '__main__':
    # Load in data
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    name = ['four_systems/tro_four_systems_0', 'four_systems/tro_four_systems_sample']
    system_idx = [0,1,2,3]
    proc_df = process_files_and_get_dataframes(system_idx, name)

    # Summary of statistics
    print("mean errors: " + str(np.mean(proc_df["errors_pos"])))
    print("std errors: " + str(np.std(proc_df["errors_pos"])))
    print("success rate: " + str(proc_df[proc_df["errors_pos"] < 1].shape))

    sns.boxplot(x="system", y="errors_pos", hue='name', data=proc_df)
    plt.show()
