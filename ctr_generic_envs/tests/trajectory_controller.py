import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from trajectory_plotter import load_agent, plot_trajectory
from trajectory_generator import line_traj, circle_traj, polygon_traj, helix_traj
from trajectory_animation import animate_trajectory

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def trajectory_controller(agent, env, x_traj, y_traj, z_traj):
    obs = env.reset()
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    for i in range(0, len(x_traj)):
        goal = np.array([x_traj[i], y_traj[i], z_traj[i]])
        obs = env.env.reset(goal=goal)
        # Set desired goal as x,y,z trajectory point in obs
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)
            obs_dict = env.convert_obs_to_dict(obs)
            achieved_goals.append(obs_dict['achieved_goal'])
            desired_goals.append(obs_dict['desired_goal'])
            r1.append(env.env.model.r1)
            r2.append(env.env.model.r2)
            r3.append(env.env.model.r3)
            # After each step, store achieved goal as well as rs
            if done or infos.get('is_success', False):
                break
    return achieved_goals, desired_goals, r1, r2, r3


if __name__ == '__main__':
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/free_rot/p_256_units/her/CTR-Generic-Reach-v0_1/best_model.zip"
    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'num_systems': 1,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    # Get trajectory points
    x_points, y_points, z_points = line_traj(10, 0.20, 0, 0.20, 0, 0.20, 0.20)
    # Run through trajectory controller and save goals and shape
    achieved_goals, desired_goals, r1, r2, r3 = trajectory_controller(model, env, x_points, y_points, z_points)

    # Plot the full trajectory
    #fig, ax = plot_trajectory(achieved_goals, desired_goals, r1, r2, r3)
    #ax.plot3D(x_points, y_points, z_points, marker='.', linestyle=':' )
    #plt.show()

    # Animate full trajectory
    # TODO: Animate doesn't plot trajectory but rather single goal
    animate_trajectory(achieved_goals, desired_goals, r1, r2, r3)
    #ax.plot3D(x_points, y_points, z_points, marker='.', linestyle=':')
