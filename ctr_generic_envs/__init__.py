from gym.envs.registration import register
import numpy as np

register(
    id='CTR-Generic-Reach-v0', entry_point='ctr_generic_envs.envs:CtrGenericEnv',
    kwargs={
        'ctr_systems': {
            'ctr_0': {
                # Autonomous steering by Mohsen Khadem
                # 'tube_0':
                #    {'length': 431e-3, 'length_curved': 103e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 1.10e-3,
                #     'stiffness': 10.25e+10, 'torsional_stiffness': 18.79e+10, 'x_curvature': 21.3, 'y_curvature': 0
                #     },

                # 'tube_1':
                #    {'length': 332e-3, 'length_curved': 113e-3, 'diameter_inner': 1.4e-3, 'diameter_outer': 1.8e-3,
                #     'stiffness': 68.6e+10, 'torsional_stiffness': 11.53e+10, 'x_curvature': 13.1, 'y_curvature': 0
                #     },

                # 'tube_2':
                #    {'length': 174e-3, 'length_curved': 134e-3, 'diameter_inner': 2e-3, 'diameter_outer': 2.4e-3,
                #     'stiffness': 16.96e+10, 'torsional_stiffness': 14.25e+10, 'x_curvature': 3.5, 'y_curvature': 0
                #     }
                # Unknown tube parameters or where they are from
                'tube_0':
                    {'length': 215e-3, 'length_curved': 14.9e-3, 'diameter_inner': 1.0e-3, 'diameter_outer': 2.4e-3,
                     'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0,
                     },

                'tube_1':
                    {'length': 120.2e-3, 'length_curved': 21.6e-3, 'diameter_inner': 3.0e-3, 'diameter_outer': 3.8e-3,
                     'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0,
                    },

                'tube_2':
                    {'length': 48.5e-3, 'length_curved': 8.8e-3, 'diameter_inner': 4.4e-3, 'diameter_outer': 5.4e-3,
                     'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0,
                    }
            },

        },
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
        'initial_q': [0, 0, 0, 0, 0, 0],
        'relative_q': False,
        'resample_joints': False,
        'render': False,
        'evaluation': False
    },
    max_episode_steps=150
)
