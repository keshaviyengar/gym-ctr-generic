# Iterate through models
# Complete a path following task
# Save the frames, include timestep number
# Go to next model and repeat

import numpy as np
from moviepy.editor import *

from trajectory_plotter import load_agent, plot_trajectory, plot_path_only, plot_intermediate, run_episode
from trajectory_generator import line_traj, circle_traj, polygon_traj, helix_traj
from trajectory_animation import animate_trajectory


create_animations = True
combine_mp4 = True

if __name__ == '__main__':
    file_names = list()
    for i in range(200000, 3000000+1, 200000):
        # Load in model
        gen_model_path = "/her/CTR-Generic-Reach-v0_1/rl_model_" + str(i) + "_steps.zip"
        #gen_model_path = "/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0.zip"
        project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
        name = 'free_rotation/tro_free_0'
        selected_systems = [0]
        model_path = project_folder + name + gen_model_path
        output_path = project_folder + name

        env_id = "CTR-Generic-Reach-v0"
        env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'constrain_alpha': False,
                      'num_systems': len(selected_systems), 'select_systems': selected_systems,
                      'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                    'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001},
                      }
        try:
            env, model = load_agent(env_id, env_kwargs, model_path)
            file_name = '/home/keshav/test_' + str(i) + '.mp4'
        except:
            continue
        file_names.append('/home/keshav/test_' + str(i) + '.mp4')
        if create_animations:
            # Run a episode reaching a goal
            achieved_goals, desired_goals, r1, r2, r3 = run_episode(env, model)
            ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, training_step=str(i))
            ani.save(file_name, writer='imagemagick', fps=5)

    clips = []
    for i, file_name in enumerate(file_names):
        clip = VideoFileClip(file_name)
        clips.append(clip)

    clips_combined = concatenate_videoclips(clips)
    clips_combined.write_videofile('/home/keshav/test_merged.mp4')
