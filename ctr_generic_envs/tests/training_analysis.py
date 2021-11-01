import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def process_files_and_get_dataframes(all_files):
    dfs = []
    names = ['5_million']
    for f, name in zip(all_files, names):
        df = pd.read_csv(f)
        df["exp"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_errors_and_success_rate(df):
    sns.lineplot(x='total/steps', y='errors', data=df, hue='exp')
    ax2 = plt.twinx()
    sns.lineplot(x='total/steps', y='success rate', data=df, hue='exp', legend=False, palette=['red'])
    plt.show()


if __name__ == '__main__':
    # Load in progress.csv file with data to plot for each experiment
    all_files = ['/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/generic_policy/mk_system/tro_1_million_generic_1/progress.csv',
                 '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/generic_policy/mk_system/tro_2_million_generic_1/progress.csv']
    proc_df = process_files_and_get_dataframes(all_files)

    plot_error_and_success = True
    plot_goal_tolerance = False
    if plot_error_and_success:
        plot_errors_and_success_rate(proc_df)
    if plot_goal_tolerance:
        sns.lineplot(x='total/steps', y='rollout/goal_tolerance', data=proc_df, hue='exp')
        plt.show()

