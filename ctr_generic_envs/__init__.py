from gym.envs.registration import register
import numpy as np

register(
    id='CTR-Generic-Reach-v0', entry_point='ctr_generic_envs.envs:CtrGenericEnv',
    kwargs={
        # Ranges
        # L = [10e-3, 500.0e-3]
        # L_c = [10.0e-3, 500.0e-3]
        # d_i = [0.1e-3, 2.0e-3]
        # d_o = [0.1e-3, 2.0e-3]
        # E_I = [5.0e+9, 50.0e+10]
        # G_J = [1.0e+10, 30.0e+10]
        # x_curv = [1.0, 25.0]
        'tube_parameters_min': {'L': 10e-3, 'L_c': 10.0e-3, 'd_i': 0.1e-3, 'd_o': 0.1e-3, 'E_I': 5.0e+9,
                                'G_J': 1.0e+10, 'x_curv': 1.0},
        'tube_parameters_max': {'L': 500e-3, 'L_c': 500.0e-3, 'd_i': 2.0e-3, 'd_o': 2.0e-3, 'E_I': 50.0e+10,
                                'G_J': 30.0e+10, 'x_curv': 25.0},
        'action_length_limit': 0.001,
        'action_rotation_limit': 5,
        'max_episode_steps': 150,
        'n_substeps': 10,
        'goal_tolerance_parameters': {
            'inc_tol_obs': False, 'initial_tol': 0.020, 'final_tol': 0.001,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0
        },
        'noise_parameters': {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        },
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        'initial_q': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'resample_joints': False,
        'render': False,
        'evaluation': False,
        'domain_rand': 0.0
    },
    max_episode_steps=150
)
