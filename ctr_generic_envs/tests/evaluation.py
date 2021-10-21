import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# Aim of this script is to run through a number of episodes, returns the error statistics

def evaluation(env_id, exp_id, model_path, num_episodes, output_path):
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
    model = HER.load(model_path, env=env)

    seed = np.random.randint(0, 10)
    set_global_seeds(seed)

    goal_errors = np.empty((num_episodes), dtype=float)
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
    system_idx = np.empty((num_episodes, 1), dtype=float)

    for episode in range(num_episodes):
        print('episode: ', episode)
        # Run random episodes and save sequence of actions and states to plot in matlab
        episode_reward = 0
        ep_len = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)

            episode_reward += reward
            ep_len += 1

            if done or infos.get('is_success', False):
                goal_errors[episode] = infos.get('errors_pos')
                q_B_desired[episode, :] = infos.get('q_desired')[:3]
                q_alpha_desired[episode, :] = infos.get('q_desired')[3:]
                q_B_achieved[episode, :] = infos.get('q_achieved')[:3]
                q_alpha_achieved[episode, :] = infos.get('q_achieved')[3:]
                desired_goals[episode, :] = infos.get('desired_goal')
                achieved_goals[episode, :] = infos.get('achieved_goal')
                starting_positions[episode, :] = infos.get('starting_position')
                q_B_starting[episode, :] = infos.get('q_starting')[:3]
                q_alpha_starting[episode, :] = infos.get('q_starting')[3:]
                system_idx[episode, :] = infos.get('system_idx')
                print("error: ", infos.get("errors_pos"))
                break

    print('mean_errors: ', np.mean(goal_errors))
    eval_df = pd.DataFrame(data=np.column_stack((desired_goals, achieved_goals, starting_positions,
                                                 q_B_desired, q_B_achieved, q_B_starting, q_alpha_desired,
                                                 q_alpha_achieved, q_alpha_starting)),
                           columns=['desired_goal_x', 'desired_goal_y', 'desired_goal_z',
                                    'achieved_goal_x', 'achieved_goal_y', 'achieved_goal_z',
                                    'starting_position_x', 'starting_position_y', 'starting_position_z',
                                    'B_desired_1', 'B_desired_2', 'B_desired_3',
                                    'B_achieved_1', 'B_achieved_2', 'B_achieved_3',
                                    'B_starting_1', 'B_starting_2', 'B_starting_3',
                                    'alpha_desired_1', 'alpha_desired_2', 'alpha_desired_3',
                                    'alpha_achieved_1', 'alpha_achieved_2', 'alpha_achieved_3',
                                    'alpha_startin_1', 'alpha_starting_2', 'alpha_starting_3',
                                    ])
    eval_df.to_csv(output_path)


if __name__ == '__main__':
    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    num_episodes = 1000
    experiment_ids = ["tro_1_million_generic_1", "tro_2_million_generic_1", "tro_5_million_generic_1"]
    for exp_id in experiment_ids:
        model_path = "/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/generic_policy/mk_system/" + exp_id + "/her/CTR-Generic-Reach-v0_1/best_model.zip"
        output_path = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/generic_policy/data/mk_system/' + exp_id + '_error_analysis.csv'
        evaluation(env_id, exp_id, model_path, num_episodes, output_path)
