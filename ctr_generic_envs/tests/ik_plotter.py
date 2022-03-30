import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from trajectory_plotter import load_agent, plot_trajectory, plot_path_only, plot_intermediate, run_episode
from trajectory_animation import animate_trajectory

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Script takes an environment for CTRs and plots the followed trajectory with the CTR system
# 1. Load in agent
# 2. Run an episode
# 3. Track the achieved goals and desired goal
# 4. Plot the trajectory taken and plot

if __name__ == '__main__':
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0.zip"
    #gen_model_path = "/her/CTR-Generic-Reach-v0_1/rl_model.zip"

    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/old_rotation_experiments/rotation_experiments/'
    name = 'free_rotation/tro_free_0'
    selected_systems = [0]

    model_path = project_folder + name + gen_model_path
    output_path = project_folder + name

    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'constrain_alpha': False,
                  'num_systems': len(selected_systems), 'select_systems': selected_systems,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    goal = np.array([0.0, 0.05, 0.15])
    ags, dg, r1, r2, r3 = run_episode(env, model, goal)
    plot_trajectory(ags, dg, r1, r2, r3, save_path='/home/keshav/tro_free_ik_1.png')
    ags, dg, r1, r2, r3 = run_episode(env, model, goal)
    plot_trajectory(ags, dg, r1, r2, r3, save_path='/home/keshav/tro_free_ik_2.png')

