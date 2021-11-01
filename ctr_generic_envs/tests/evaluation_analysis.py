import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def process_files_and_get_dataframes(all_files):
    dfs = []
    names = ['1_million', '2_million']
    for f, name in zip(all_files, names):
        df = pd.read_csv(f)
        dg = np.array([df['desired_goal_x'], df['desired_goal_y'], df['desired_goal_z']])
        ag = np.array([df['achieved_goal_x'], df['achieved_goal_y'], df['achieved_goal_z']])
        sg = np.array([df['starting_position_x'], df['starting_position_y'], df['starting_position_z']])
        df['errors_pos'] = np.linalg.norm(np.transpose(ag - dg), axis=1) * 1000
        df['goal_dist'] = np.linalg.norm(np.transpose(sg - dg), axis=1) * 1000
        df["exp"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_alpha_box_plots(df, alpha):
    sns.boxplot(x="alpha_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Alpha " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.show()

def plot_B_box_plots(df, alpha):
    sns.boxplot(x="B_achieved_" + str(alpha) + "_bins", y="errors_pos", data=df)
    plt.xlabel("Beta " + str(alpha) + " achieved joint positions (radians)")
    plt.ylabel("Errors (mm)")
    plt.show()


if __name__ == '__main__':
    all_files = [
        '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/generic_policy/data/mk_system/tro_5_million_generic_1_error_analysis.csv']
    proc_df = process_files_and_get_dataframes(all_files)

    plot_goal_distance_scatter = False
    plot_rot_joints_box_plot = False
    plot_ext_joints_box_plot = False
    plot_3d_workspace = True

    if plot_goal_distance_scatter:
        sns.regplot(x='goal_dist', y='errors_pos', data=proc_df, ci=None, scatter_kws={"s": 10})
        plt.show()

    num_bins = 20
    if plot_rot_joints_box_plot:
        # Rotation plots
        rotation_bins = np.linspace(np.deg2rad(-180), np.deg2rad(180), num_bins)
        rot_label_bins = np.around(np.linspace(np.deg2rad(-180), np.deg2rad(180), num_bins - 1), 2)
        for alpha in range(1, 4):
            proc_df["alpha_achieved_" + str(alpha)] = (proc_df["alpha_achieved_" + str(alpha)] + np.pi) % (
                        2 * np.pi) - np.pi
            proc_df["alpha_achieved_" + str(alpha) + "_bins"] = pd.cut(proc_df["alpha_achieved_" + str(alpha)],
                                                                       bins=rotation_bins, labels=rot_label_bins)
            plot_alpha_box_plots(proc_df, alpha)

    if plot_ext_joints_box_plot:
        # Extension plots
        min_beta = np.min([proc_df.B_achieved_1.min(), proc_df.B_achieved_2.min(), proc_df.B_achieved_3.min()])
        max_beta = np.max([proc_df.B_achieved_1.max(), proc_df.B_achieved_2.max(), proc_df.B_achieved_3.max()])
        extension_bins = np.linspace(min_beta, max_beta, num_bins)
        ext_label_bins = np.around(np.linspace(min_beta, max_beta, num_bins - 1), 2)
        for beta in range(1, 4):
            proc_df["B_achieved_" + str(beta) + "_bins"] = pd.cut(proc_df["B_achieved_" + str(beta)],
                                                                     bins=extension_bins, labels=ext_label_bins)
            plot_B_box_plots(proc_df, beta)

    # 3D workspace plots
    if plot_3d_workspace:
        fig = plt.figure()
        ax3D = fig.add_subplot(1, 2, 1, projection='3d')
        ax3D.scatter(proc_df['achieved_goal_x'] * 1000, proc_df['achieved_goal_y'] * 1000, proc_df['achieved_goal_z'] * 1000,
                     c=proc_df['errors_pos'])
        ax3D.set_xlabel("X (mm)")
        ax3D.set_ylabel("Y (mm)")
        ax3D.set_zlabel("Z (mm)")
        ax3D.set_title("Achieved goals with Errors")
        ax3D = fig.add_subplot(1, 2, 2, projection='3d')
        proc_df_tol = proc_df[proc_df['errors_pos'] > 5]
        p = ax3D.scatter(proc_df_tol['achieved_goal_x'] * 1000, proc_df_tol['achieved_goal_y'] * 1000,
                         proc_df_tol['achieved_goal_z'] * 1000,
                     c=proc_df_tol['errors_pos'])
        ax3D.set_xlabel("X (mm)")
        ax3D.set_ylabel("Y (mm)")
        ax3D.set_zlabel("Z (mm)")
        ax3D.set_title("Achieved goals with Errors > 5 mm")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.5])
        fig.colorbar(p, cax=cbar_ax)
        plt.show()
