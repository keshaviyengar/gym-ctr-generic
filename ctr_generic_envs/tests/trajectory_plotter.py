import gym
import ctr_generic_envs
import numpy as np

from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper
from stable_baselines.bench.monitor import Monitor

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Script takes an environment for CTRs and plots the followed trajectory with the CTR system
# 1. Load in agent
# 2. Run an episode
# 3. Track desired, achieved goals as well as robot shape
# 4. Plot tracked statistics


def load_agent(env_id, env_kwargs, model_path, seed=None):
    if seed is None:
        seed = np.random.randint(0, 10)
        set_global_seeds(seed)
    env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
    model = HER.load(model_path, env=env)
    return env, model

def run_episode(env, model, goal=None):
    if goal is not None:
        obs = env.reset(**{'goal': goal})
    else:
        obs = env.reset()
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
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
            print("Tip Error: " + str(infos.get('errors_pos')*1000))
            print("Achieved joint: " + str(np.rad2deg(infos.get('q_achieved'))))
            break
    return achieved_goals, desired_goals, r1, r2, r3

def plot_trajectory(achived_goals, desired_goals, r1, r2, r3, save_path=None):
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.axes(projection='3d')
    ax.plot3D(r1[0][:,0] * 1000, r1[0][:,1] * 1000, r1[0][:,2] * 1000, linewidth=2.0, c='blue')
    ax.plot3D(r2[0][:,0] * 1000, r2[0][:,1] * 1000, r2[0][:,2] * 1000, linewidth=3.0, c='red')
    ax.plot3D(r3[0][:,0] * 1000, r3[0][:,1] * 1000, r3[0][:,2] * 1000, linewidth=4.0, c='green')
    ag = np.array(achived_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(ag[:,0], ag[:,1], ag[:,2], marker='.', linestyle=':', label='achieved', c='black')
    ax.scatter(ag[0,0], ag[0,1], ag[0,2], c='red', linewidth=5.0)
    ax.scatter(dg[0,0], dg[0,1], dg[0,2], c='green', linewidth=5.0)
    ax.plot3D(r1[-1][:,0] * 1000, r1[-1][:,1] * 1000, r1[-1][:,2] * 1000, linewidth=2.0, c='blue')
    ax.plot3D(r2[-1][:,0] * 1000, r2[-1][:,1] * 1000, r2[-1][:,2] * 1000, linewidth=3.0, c='red')
    ax.plot3D(r3[-1][:,0] * 1000, r3[-1][:,1] * 1000, r3[-1][:,2] * 1000, linewidth=4.0, c='green')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim3d([-100, 100])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zlim3d([0.0, 200])
    ax.set_zticks([0, 50, 100, 150, 200])
    #ax.set_xlim3d([-50, 50])
    #ax.set_xticks([-50, 0, 50])
    #ax.set_ylim3d([-50, 50])
    #ax.set_yticks([-50, 0, 50])
    #ax.set_zlim3d([0.0, 100])
    #ax.set_zticks([0, 50, 100])
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

def plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, traj_point, save_path=None):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.axes(projection='3d')
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(r1[traj_point][:,0] * 1000, r1[traj_point][:,1] * 1000, r1[traj_point][:,2] * 1000, linewidth=2.0, c='blue')
    ax.plot3D(r2[traj_point][:,0] * 1000, r2[traj_point][:,1] * 1000, r2[traj_point][:,2] * 1000, linewidth=3.0, c='red')
    ax.plot3D(r3[traj_point][:,0] * 1000, r3[traj_point][:,1] * 1000, r3[traj_point][:,2] * 1000, linewidth=4.0, c='green')
    ax.plot3D(ag[:traj_point,0], ag[:traj_point,1], ag[:traj_point,2], marker='.', linestyle='-', label='achieved', c='purple')
    ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='desired', c='black')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim3d([0, 100])
    ax.set_xticks([0, 50, 100])
    ax.set_ylim3d([0, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_zlim3d([0.0, 250])
    ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

def plot_path_only(achieved_goals, desired_goals, save_path=None):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.axes(projection='3d')
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(ag[:,0], ag[:,1], ag[:,2], marker='.', linestyle='-', label='achieved', c='purple')
    ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='desired', c='black')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim3d([30, 90])
    ax.set_xticks([30, 50, 70, 90])
    ax.set_ylim3d([30, 90])
    ax.set_yticks([30, 50, 70, 90])
    ax.set_zlim3d([170, 260])
    ax.set_zticks([170, 200, 230, 270])
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

if __name__ == '__main__':
    model_path = "/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/free_rot/p_256_units/her/CTR-Generic-Reach-v0_1/best_model.zip"
    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'num_systems': 1,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)
    achieved_goals, desired_goals, r1, r2, r3 = run_episode(env, model)
    fig, ax = plot_trajectory(achieved_goals, desired_goals, r1, r2, r3)
    plt.show()
