import gym
import ctr_generic_envs
import numpy as np

from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Script takes an environment for CTRs and plots the followed trajectory with the CTR system
# 1. Load in agent
# 2. Run an episode
# 3. Track desired, achieved goals as well as robot shape
# 4. Create an animation of the robot

def load_agent(env_id, env_kwargs, model_path, seed=None):
    if seed is None:
        seed = np.random.randint(0, 10)
        set_global_seeds(seed)
    env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
    model = HER.load(model_path, env=env)
    return env, model

def run_episode(env, agent):
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
            break
    return achieved_goals, desired_goals, r1, r2, r3

def update_animation(num, tube1, tube2, tube3, ag_p, achieved_goals, r1, r2, r3):
    tube1.set_data(r1[num][:,0], r1[num][:,1])
    tube1.set_3d_properties(r1[num][:,2])

    tube2.set_data(r2[num][:,0], r2[num][:,1])
    tube2.set_3d_properties(r2[num][:,2])

    tube3.set_data(r3[num][:,0], r3[num][:,1])
    tube3.set_3d_properties(r3[num][:,2])

    ag_p.set_data(achieved_goals[:num,0], achieved_goals[:num,1])
    ag_p.set_3d_properties(achieved_goals[:num,2])
    return [tube1, tube2, tube3, ag_p]


def animate_trajectory(achieved_goals, desired_goals, r1, r2 ,r3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    axis_lims = 0.15
    ax.set_xlim3d([-axis_lims, axis_lims])
    ax.set_xlabel('X')

    ax.set_ylim3d([-axis_lims, axis_lims])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 2 * axis_lims])
    ax.set_zlabel('Z')
    ag = np.array(achieved_goals)
    dg = np.array(desired_goals)
    tube1, = ax.plot3D(r1[0][:, 0], r1[0][:, 1], r1[0][:, 2], linewidth=2.0)
    tube2, = ax.plot3D(r2[0][:, 0], r2[0][:, 1], r2[0][:, 2], linewidth=3.0)
    tube3, = ax.plot3D(r3[0][:, 0], r3[0][:, 1], r3[0][:, 2], linewidth=4.0)
    ag_p, = ax.plot3D(ag[0, 0], ag[0, 1], ag[0, 2], marker='.', linestyle=':')
    ax.plot3D(dg[0, 0], dg[0, 1], dg[0, 2], color='green', marker='o')

    ani = animation.FuncAnimation(fig, update_animation, len(achieved_goals), fargs=[tube1, tube2, tube3, ag_p, ag, r1, r2, r3])
    plt.show()

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
    animate_trajectory(achieved_goals, desired_goals, r1, r2, r3)
