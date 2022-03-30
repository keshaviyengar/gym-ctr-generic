import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from trajectory_plotter import load_agent, plot_trajectory, plot_path_only, plot_intermediate
from trajectory_generator import line_traj, circle_traj, polygon_traj, helix_traj
from trajectory_animation import animate_trajectory

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def trajectory_controller(agent, env, x_traj, y_traj, z_traj, system_idx, select_systems):
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    # Get to first point in trajectory then start recording
    goal = np.array([x_traj[0], y_traj[0], z_traj[0]])
    if len(select_systems) > 1:
        obs = env.reset(**{'system_idx': np.where(system_idx == np.array(select_systems))[0][0],
                           'goal': goal})
    else:
        obs = env.reset(**{'goal': goal})
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        obs_dict = env.convert_obs_to_dict(obs)
        # After each step, store achieved goal as well as rs
        if done or infos.get('is_success', False):
            break

    achieved_goals.append(obs_dict['achieved_goal'])
    desired_goals.append(obs_dict['desired_goal'])
    r1.append(env.env.model.r1)
    r2.append(env.env.model.r2)
    r3.append(env.env.model.r3)

    for i in range(1, len(x_traj)):
        goal = np.array([x_traj[i], y_traj[i], z_traj[i]])
        if len(select_systems) > 1:
            obs = env.reset(**{'system_idx': np.where(system_idx == np.array(select_systems))[0][0],
                               'goal': goal})
        else:
            obs = env.reset(**{'goal': goal})
        # Set desired goal as x,y,z trajectory point in obs
        print(str(i) + ' out of ' + str(len(x_traj)))
        for _ in range(20):
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
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0.zip"
    #gen_model_path = "/her/CTR-Generic-Reach-v0_1/best_model.zip"

    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/old_rotation_experiments/rotation_experiments/'
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
    #name = 'four_systems/tro_four_systems_sample'
    name = 'free_rotation/tro_free_0'
    selected_systems = [0]

    model_path = project_folder + name + gen_model_path
    output_path = project_folder + name

    noisy_env = False
    if noisy_env:
        noise_parameters =  {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(1.0), 'extension_std': 0.001 * np.deg2rad(1.0), 'tracking_std': 0.0008
        }
    else:
        noise_parameters =  {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        }

    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': False, 'constrain_alpha': False,
                  'num_systems': len(selected_systems), 'select_systems': selected_systems,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001},
                  'noise_parameters': noise_parameters,
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    # Get trajectory points
    #x_points, y_points, z_points = line_traj(20, 0.05, 0.05, 0.20, 0.1, 0.05, 0.20)
    #x_points, y_points, z_points = line_traj(20, 0.0, 0.10, 0.20, 0.10, 0.10, 0.20)
    #x_points, y_points, z_points = line_traj(20, -0.025, 0.0, 0.15, 0.025, 0.0, 0.15)
    #x_points,y_points,z_points = helix_traj(100, 3, 0.03, 0.005, [0.06,0.06,0.18])
    x_points, y_points, z_points = circle_traj(40, 0.0, 0.10, 0.20, 0.05)
    # Run through trajectory controller and save goals and shape
    achieved_goals, desired_goals, r1, r2, r3 = trajectory_controller(model, env, x_points, y_points, z_points, 0, selected_systems)

    errors = np.linalg.norm(np.array(achieved_goals) - np.array(desired_goals), axis=1)
    print('mean tracking error: ' + str(np.mean(errors) * 1000))
    print('std tracking error: ' + str(np.std(errors) * 1000))
    # Plot three trajectories intermediate
    #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, 0, '/home/keshav/cras_straight_1.png')
    #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, int(len(achieved_goals) / 2), '/home/keshav/cras_straight_2.png')
    #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, int(len(achieved_goals) - 1))

    # Plot the path only
    #plot_path_only(achieved_goals, desired_goals)

    # Animate full trajectory
    # TODO: Animate doesn't plot trajectory but rather single goal
    ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, title=False)
    ani.save(output_path + '_circle.gif', writer='imagemagick', fps=5)
