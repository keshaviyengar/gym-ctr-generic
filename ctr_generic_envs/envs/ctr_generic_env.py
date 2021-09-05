import gym
import numpy as np

from ctr_generic_envs.envs.trig_obs import TrigObs
from ctr_generic_envs.envs.exact_model import ExactModel

from ctr_generic_envs.envs.CTR_Python import Tube


class TubeParameters(object):
    def __init__(self, length, length_curved, outer_diameter, inner_diameter, stiffness, torsional_stiffness,
                 x_curvature, y_curvature):
        self.L = length
        self.L_s = length - length_curved
        self.L_c = length_curved
        # Exact model
        self.J = (np.pi * (pow(outer_diameter, 4) - pow(inner_diameter, 4))) / 32
        self.I = (np.pi * (pow(outer_diameter, 4) - pow(inner_diameter, 4))) / 64
        self.E = stiffness
        self.G = torsional_stiffness
        self.U_x = x_curvature
        self.U_y = y_curvature


class GoalTolerance(object):
    def __init__(self, goal_tolerance_parameters):
        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.inc_tol_obs = self.goal_tolerance_parameters['inc_tol_obs']
        self.init_tol = self.goal_tolerance_parameters['initial_tol']
        self.final_tol = self.goal_tolerance_parameters['final_tol']
        self.N_ts = self.goal_tolerance_parameters['N_ts']
        self.function = self.goal_tolerance_parameters['function']
        valid_functions = ['constant', 'linear', 'decay']
        if self.function not in valid_functions:
            print('Not a valid function, defaulting to constant')
            self.function = 'constant'

        if self.function == 'constant':
            self.init_tol = self.final_tol

        if self.function == 'linear':
            self.a = (self.final_tol - self.init_tol) / self.N_ts
            self.b = self.init_tol

        if self.function == 'decay':
            self.a = self.init_tol
            self.r = 1 - np.power((self.final_tol / self.init_tol), 1 / self.N_ts)

        self.set_tol = self.goal_tolerance_parameters['set_tol']
        if self.set_tol == 0:
            self.current_tol = self.init_tol
        else:
            self.current_tol = self.set_tol
        self.training_step = 0

    def update(self):
        if self.set_tol == 0:
            if (self.function == 'linear') and (self.training_step <= self.N_ts):
                self.current_tol = self.linear_function()
            elif (self.function == 'decay') and (self.training_step <= self.N_ts):
                self.current_tol = self.decay_function()
            else:
                self.current_tol = self.final_tol
            self.training_step += 1
        else:
            self.current_tol = self.set_tol

    def get_tol(self):
        return self.current_tol

    def linear_function(self):
        return self.a * self.training_step + self.b

    def decay_function(self):
        return self.a * np.power(1 - self.r, self.training_step)


class CtrGenericEnv(gym.GoalEnv):
    def __init__(self, ctr_systems, action_length_limit, action_rotation_limit, max_episode_steps, n_substeps,
                 goal_tolerance_parameters, noise_parameters, relative_q, initial_q, resample_joints, render,
                 evaluation):
        self.num_systems = len(ctr_systems.keys())
        self.systems = list()
        for i in range(0, self.num_systems):
            system_args = ctr_systems['ctr_' + str(i)]
            # Extract tube parameters
            num_tubes = len(system_args.keys())
            tubes = list()
            for i in range(0, num_tubes):
                tube_args = system_args['tube_' + str(i)]
                tubes.append(Tube(**tube_args))
            self.systems.append(tubes)

        self.num_tubes = len(self.systems[0])

        self.action_length_limit = action_length_limit
        self.action_rotation_limit = action_rotation_limit
        # Action space
        action_length_limit = np.full(self.num_tubes, self.action_length_limit)
        action_orientation_limit = np.full(self.num_tubes, np.deg2rad(self.action_rotation_limit))
        self.action_space = gym.spaces.Box(low=np.concatenate((-action_length_limit, -action_orientation_limit)),
                                           high=np.concatenate((action_length_limit, action_orientation_limit)),
                                           dtype="float32")

        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self.resample_joints = resample_joints
        self.desired_q = []

        self.model = ExactModel(self.systems)
        ext_tol = 1e-4
        self.r_df = None

        self.rep_obj = TrigObs(self.systems, goal_tolerance_parameters, noise_parameters, initial_q, relative_q, ext_tol)
        self.goal_tol_obj = GoalTolerance(goal_tolerance_parameters)
        self.t = 0
        self.evaluation = evaluation
        self.observation_space = self.rep_obj.get_observation_space()

        self.system_idx = 0

    # TODO: Reset the system idx
    def reset(self, goal=None):
        self.t = 0
        self.system_idx = 0
        self.r_df = None
        if goal is None:
            # Resample a desired goal and its associated q joint
            self.desired_q = self.rep_obj.sample_goal(self.system_idx)
            desired_goal = self.model.forward_kinematics(self.desired_q, self.system_idx)
        else:
            desired_goal = goal
        if self.resample_joints:
            self.starting_joints = self.rep_obj.sample_goal(self.system_idx)
            self.rep_obj.set_q(self.starting_joints)
            achieved_goal = self.model.forward_kinematics(self.rep_obj.get_q(), self.system_idx)
            self.starting_position = achieved_goal
        else:
            achieved_goal = self.model.forward_kinematics(self.rep_obj.get_q(), self.system_idx)
            self.starting_position = achieved_goal
            self.starting_joints = self.rep_obj.get_q()
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol(), self.system_idx)
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        # Update goal tolerance value
        self.goal_tol_obj.update()
        for _ in range(self.n_substeps):
            self.rep_obj.set_action(action, self.system_idx)
        # Compute FK
        achieved_goal = self.model.forward_kinematics(self.rep_obj.q, self.system_idx)
        desired_goal = self.rep_obj.get_desired_goal()
        self.t += 1
        reward = self.compute_reward(achieved_goal, desired_goal, dict())
        done = (reward == 0) or (self.t >= self.max_episode_steps)
        obs = self.rep_obj.get_obs(desired_goal, achieved_goal, self.goal_tol_obj.get_tol(), self.system_idx)
        if self.evaluation:
            # Evaluation infos
            info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
                    'error': np.linalg.norm(desired_goal - achieved_goal)}
            #info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
            #        'errors_pos': np.linalg.norm(desired_goal - achieved_goal),
            #        'errors_orient': 0,
            #        'position_tolerance': self.goal_tol_obj.get_tol(),
            #        'orientation_tolerance': 0,
            #        'achieved_goal': achieved_goal,
            #        'desired_goal': desired_goal, 'starting_position': self.starting_position,
            #        'q_desired': self.desired_q, 'q_achieved': self.rep_obj.get_q(), 'q_starting': self.starting_joints}
        else:
            info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
                    'error': np.linalg.norm(desired_goal - achieved_goal)}
            #info = {'is_success': (np.linalg.norm(desired_goal - achieved_goal) < self.goal_tol_obj.get_tol()),
            #        'errors_pos':  np.linalg.norm(desired_goal - achieved_goal),
            #        'errors_orient': 0,
            #        'position_tolerance': self.goal_tol_obj.get_tol(),
            #        'orientation_tolerance': 0}

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.goal_tol_obj.get_tol()).astype(np.float64)

    def close(self):
        print("Closed env.")

    def update_goal_tolerance(self):
        self.goal_tol_obj.update()

    def get_goal_tolerance(self):
        return self.goal_tol_obj.get_tol()

    def print_parameters(self):
        print("----Observation and q_space----")
        print("relative q: ", self.rep_obj.relative_q)
        print("tolerance params: ", self.rep_obj.goal_tolerance_parameters)

        print("----Goal tolerance parameters----")
        print("init, final tol: ", self.goal_tol_obj.init_tol, ", ", self.goal_tol_obj.final_tol)
        print("N_ts: ", self.goal_tol_obj.N_ts)
        print("tolerance function: ", self.goal_tol_obj.function)

    def get_obs_dim(self):
        return self.rep_obj.obs_dim

    def get_goal_dim(self):
        return self.rep_obj.goal_dim
