import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from stable_baselines.bench.monitor import Monitor

from utils import load_agent
from plotting_utils import plot_trajectory


# Aim of this script is to run through a number of episodes, returns the error statistics. This script is used for generic
# policy evaluation and iterates through each system idx.

def evaluation(env, model, num_episodes, output_path, system):
    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    # Iterate through all systems that have been generalized
    obs = env.reset(tube_params=system)
    for i in range(1):
        goal_errors = np.empty((num_episodes), dtype=float)
        episode_lengths = np.empty((num_episodes), dtype=float)
        B_errors = np.empty((num_episodes), dtype=float)
        alpha_errors = np.empty((num_episodes), dtype=float)
        q_B_achieved = np.empty((num_episodes, 3), dtype=float)
        q_alpha_achieved = np.empty((num_episodes, 3), dtype=float)
        q_B_desired = np.empty((num_episodes, 3), dtype=float)
        q_alpha_desired = np.empty((num_episodes, 3), dtype=float)
        desired_goals = np.empty((num_episodes, 3), dtype=float)
        achieved_goals = np.empty((num_episodes, 3), dtype=float)
        starting_positions = np.empty((num_episodes, 3), dtype=float)
        q_B_starting = np.empty((num_episodes, 3), dtype=float)
        q_alpha_starting = np.empty((num_episodes, 3), dtype=float)
        system_id = np.empty((num_episodes), dtype=int)
        for episode in range(num_episodes):
            print('episode: ', episode)
            # Run random episodes and save sequence of actions and states to plot in matlab
            episode_reward = 0
            ep_len = 0
            obs = env.reset(tube_params=system)
            # Set system idx if not None
            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, infos = env.step(action)

                episode_reward += reward
                ep_len += 1

                if done or infos.get('is_success', False):
                    goal_errors[episode] = infos.get('errors_pos')
                    episode_lengths[episode] = ep_len
                    q_B_desired[episode, :] = infos.get('q_desired')[:3]
                    q_alpha_desired[episode, :] = infos.get('q_desired')[3:]
                    q_B_achieved[episode, :] = infos.get('q_achieved')[:3]
                    q_alpha_achieved[episode, :] = infos.get('q_achieved')[3:]
                    desired_goals[episode, :] = infos.get('desired_goal')
                    achieved_goals[episode, :] = infos.get('achieved_goal')
                    starting_positions[episode, :] = infos.get('starting_position')
                    q_B_starting[episode, :] = infos.get('q_starting')[:3]
                    q_alpha_starting[episode, :] = infos.get('q_starting')[3:]
                    system_id[episode] = infos.get('system_idx')
                    print("error (mm): ", infos.get("errors_pos") * 1000)
                    print("episode_length: ", ep_len)
                    break

        print('mean_errors: ', np.mean(goal_errors))
        eval_df = pd.DataFrame(data=np.column_stack((desired_goals, achieved_goals, episode_lengths, starting_positions,
                                                     q_B_desired, q_B_achieved, q_B_starting, q_alpha_desired,
                                                     q_alpha_achieved, q_alpha_starting, system_id)),
                               columns=['desired_goal_x', 'desired_goal_y', 'desired_goal_z',
                                        'achieved_goal_x', 'achieved_goal_y', 'achieved_goal_z', 'episode_length',
                                        'starting_position_x', 'starting_position_y', 'starting_position_z',
                                        'B_desired_1', 'B_desired_2', 'B_desired_3',
                                        'B_achieved_1', 'B_achieved_2', 'B_achieved_3',
                                        'B_starting_1', 'B_starting_2', 'B_starting_3',
                                        'alpha_desired_1', 'alpha_desired_2', 'alpha_desired_3',
                                        'alpha_achieved_1', 'alpha_achieved_2', 'alpha_achieved_3',
                                        'alpha_starting_1', 'alpha_starting_2', 'alpha_starting_3', 'system_id'
                                        ])
        eval_df.to_csv(output_path + "/evaluations_" + str(0) + ".csv")


if __name__ == '__main__':
    #gen_model_path = "/her/CTR-Generic-Reach-v0_1/best_model.zip"
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/best_model.zip"

    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tmrb_2022/continuous_generalization/generalize_agent/'
    name = 'generalize_2'

    model_path = project_folder + name + gen_model_path
    output_path = project_folder + name

    num_episodes = 1000
    system_0 = [
        {'L': 0.431, 'L_c': 0.103, 'd_o': 0.00101, 'd_i': 0.0007, 'E_I': 102.5e+10, 'G_J': 187.9e+10, 'x_curv': 21.3},
        {'L': 0.332, 'L_c': 0.113, 'd_o': 0.0018, 'd_i': 0.0014, 'E_I': 685.0e+10, 'G_J': 115.3e+10, 'x_curv': 13.1},
        {'L': 0.174, 'L_c': 0.134, 'd_o': 0.0024, 'd_i': 0.002, 'E_I': 169.6e+10, 'G_J': 142.5e+10, 'x_curv': 3.5},

    ]


    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'resample_joints': False,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)
    print("output path: " + output_path)
    evaluation(env, model, num_episodes, output_path, system)
