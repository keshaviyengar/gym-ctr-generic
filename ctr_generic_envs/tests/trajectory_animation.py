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

def update_animation(num, ax, tube1, tube2, tube3, ag_p, achieved_goals, desired_goals, training_step, r1, r2, r3, title=True):
    tube1.set_data(r1[num][:,0] * 1000, r1[num][:,1] * 1000)
    tube1.set_3d_properties(r1[num][:,2] * 1000)

    tube2.set_data(r2[num][:,0] * 1000, r2[num][:,1] * 1000)
    tube2.set_3d_properties(r2[num][:,2] * 1000)

    tube3.set_data(r3[num][:,0] * 1000, r3[num][:,1] * 1000)
    tube3.set_3d_properties(r3[num][:,2] * 1000)

    ag_p.set_data(achieved_goals[:num,0], achieved_goals[:num,1])
    ag_p.set_3d_properties(achieved_goals[:num,2])

    error = np.linalg.norm(achieved_goals[num, :] - desired_goals)
    if title:
        ax.set_title('Policy: ' + str(training_step) + ' Step: ' + str(num) + '\nError: ' + str(round(error, 2)) + ' mm', fontsize=18, color='g')

    ax.view_init(elev=10., azim=num)
    return [tube1, tube2, tube3, ag_p]

def animate_trajectory(achieved_goals, desired_goals, r1, r2 ,r3, training_step=None, title=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    axis_lims = 0.15
    ax.set_box_aspect([1,1,1])
    #ax.set_xlim3d([-axis_lims, axis_lims])
    #ax.set_ylim3d([-axis_lims, axis_lims])
    #ax.set_zlim3d([0.0, 2 * axis_lims])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    tube1, = ax.plot3D(r1[0][:, 0] * 1000, r1[0][:, 1] * 1000, r1[0][:, 2] * 1000, linewidth=2.0)
    tube2, = ax.plot3D(r2[0][:, 0] * 1000, r2[0][:, 1] * 1000, r2[0][:, 2] * 1000, linewidth=3.0)
    tube3, = ax.plot3D(r3[0][:, 0] * 1000, r3[0][:, 1] * 1000, r3[0][:, 2] * 1000, linewidth=4.0)
    ag_p, = ax.plot3D(ag[0, 0], ag[0, 1], ag[0, 2], marker='.', linestyle=':')
    dg_p = ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='desired', c='black')

    ani = animation.FuncAnimation(fig, update_animation, len(achieved_goals), fargs=[ax, tube1, tube2, tube3,
                                                                                     ag_p, ag, dg, training_step, r1, r2, r3, title])
    return ani

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
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'constrain_alpha': False,
                  'num_systems': len(selected_systems), 'select_systems': selected_systems,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001},
                  'noise_parameters': noise_parameters,
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)
    achieved_goals, desired_goals, r1, r2, r3 = run_episode(env, model)
    ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, training_step='Final')
    ani.save(output_path + '_ik.gif', writer='imagemagick', fps=5)
