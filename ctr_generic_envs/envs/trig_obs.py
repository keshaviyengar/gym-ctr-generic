import numpy as np
import gym

"""
This function is the base class for the joint representations of basic, cylindrical and polar.
It should implement converting joints from basic to other representations,
setting actions and returning the resultant state and
changing the goal tolerance if variable goal tolerance is used.

q: Current joint as relative joint representation
{beta_0, beta_1 - beta_0, beta_2 - beta_1, ..., beta_{i+1} - beta_{i},  alpha_0, alpha_1 - alpha_0, alpha_2 - alpha_1,
..., alpha_{i+1} - alpha_i}
"""


# TODO: Inital q is always fully extended (all zeros)


class TrigObs(object):
    def __init__(self, tube_parameters_low, tube_parameters_high, goal_tolerance_parameters, noise_parameters,
                 initial_q):
        self.tube_lengths = list()
        self.num_tubes = 3
        self.tube_parameters_low = tube_parameters_low
        self.tube_parameters_high = tube_parameters_high

        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.noise_parameters = noise_parameters
        extension_std_noise = np.full(self.num_tubes, noise_parameters['extension_std'])
        rotation_std_noise = np.full(self.num_tubes, noise_parameters['rotation_std'])
        self.q_std_noise = np.concatenate((extension_std_noise, rotation_std_noise))
        self.tracking_std_noise = np.full(3, noise_parameters['tracking_std'])
        # Keep q as absolute joint positions, convert to relative as needed and store as absolute
        self.q = initial_q
        # TODO: Set the tube lengths
        self.tube_lengths = np.zeros(3)
        # desired, achieved goal space
        self.observation_space = self.get_observation_space()

    # TODO: Actions are -1 to 1
    def set_action(self, action):
        # TODO: Just clip the tube length vs. beta instead of using q spaces
        self.q += action
        q_betas = np.clip(self.q[:self.num_tubes], np.zeros(self.num_tubes, dtype=float), self.tube_lengths)
        q_alphas = self.q[self.num_tubes:]
        for i in range(1, self.num_tubes):
            # Remember ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            q_betas[i - 1] = min(q_betas[i - 1], q_betas[i])
            q_betas[i - 1] = max(q_betas[i - 1],
                                 self.tube_lengths[i] - self.tube_lengths[i - 1] + q_betas[i])

        self.q = np.concatenate((q_betas, q_alphas))

    # TODO: Add in alpha_U_to_alpha and beta_U_to_beta
    def sample_goal(self, tube_parameters):
        """
        Sample a joint goal while considering constraints on extension and joint limits.
        :param system: The system to to sample the goal.
        :return: Constrained achievable joint values.
        """
        # Sample a joint position
        alphas_U = np.random.uniform(low=-np.ones((1, 3)), high=np.ones((1, 3)))
        alpha_max = np.pi / 4
        alphas = np.flip(np.squeeze(self.alpha_U_to_alpha(alphas_U, alpha_max)))
        # Sample with M_B instead
        # Sample betas, ordering is reversed
        L_star = np.array([tube_parameters[2]['L'], tube_parameters[1]['L'], tube_parameters[0]['L']])
        B_U = np.random.uniform(low=-np.ones((1, 3)), high=np.ones((1, 3)))
        betas = np.flip(self.B_U_to_B(B_U, L_star[0], L_star[1], L_star[2]))
        joint_constrain = np.concatenate((betas, alphas))
        return joint_constrain

    # Conversion between normalized and un-normalized joints. Ordered outer to innermost tube.
    def B_U_to_B(self, B_U, L_1, L_2, L_3):
        B_U = np.append(B_U, 1)
        M_B = np.array([[-L_1, 0, 0],
                        [-L_1, L_1 - L_2, 0],
                        [-L_1, L_1 - L_2, L_2 - L_3]])
        normalized_B = np.block([[0.5 * M_B, 0.5 * np.matmul(M_B, np.ones((3, 1)))],
                                 [np.zeros((1, 3)), 1]])
        B = np.matmul(normalized_B, B_U)
        return B[:3]

    def alpha_U_to_alpha(self, alpha_U, alpha_max):
        return alpha_max * alpha_U

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance, tube_parameters):
        # Add noise to q, rotation and extension (encoder noise)
        noisy_q = np.random.normal(self.q, self.q_std_noise)
        # Add noise to achieved goal (tracker noise)
        noisy_achieved_goal = np.random.normal(achieved_goal, self.tracking_std_noise)
        rel_q = self.qabs2rel(noisy_q)
        rep = self.joint2rep(rel_q)
        # Normalize tube parameters
        tube_param_space = self.get_parameter_space()

        tube_params = np.array([list(tube_parameters[0].values()), list(tube_parameters[1].values()), list(tube_parameters[2].values())]).flatten()
        norm_tube_params = np.divide(tube_params, tube_param_space.high)
        if norm_tube_params.max() > 1.0 or norm_tube_params.min() < 0.0:
            print('high norm!')
        obs = np.concatenate([rep, desired_goal - noisy_achieved_goal, np.array([goal_tolerance]), norm_tube_params])
        self.obs = {
            'desired_goal': desired_goal.copy(),
            'achieved_goal': noisy_achieved_goal.copy(),
            'observation': obs.copy()
        }
        return self.obs

    # q is in relative coordinates to need to convert to absolute.
    def get_q(self):
        return self.q

    def set_q(self, q):
        self.q = q

    def qabs2rel(self, q):
        betas = q[:self.num_tubes]
        alphas = q[self.num_tubes:]
        # Compute difference
        rel_beta = np.diff(betas, prepend=0)
        rel_alpha = np.diff(alphas, prepend=0)
        return np.concatenate((rel_beta, rel_alpha))

    def qrel2abs(self, q):
        rel_beta = q[:self.num_tubes]
        rel_alpha = q[self.num_tubes:]
        betas = rel_beta.cumsum()
        alphas = rel_alpha.cumsum()
        return np.concatenate((betas, alphas))

    def get_desired_goal(self):
        return self.obs['desired_goal']

    def get_achieved_goal(self):
        return self.obs['achieved_goal']

    def get_parameter_space(self):
        diameter_diff = 0.4e-3
        parameters_low = np.array([self.tube_parameters_low['L'], self.tube_parameters_low['L_c'],
                                   self.tube_parameters_low['d_i'],
                                   self.tube_parameters_low['d_i'],
                                   self.tube_parameters_low['E_I'], self.tube_parameters_low['G_J'],
                                   self.tube_parameters_low['x_curv']])
        parameters_high = np.array([self.tube_parameters_high['L'], self.tube_parameters_high['L_c'],
                                    self.tube_parameters_high['d_o'],
                                    self.tube_parameters_high['d_o'],
                                    self.tube_parameters_high['E_I'], self.tube_parameters_high['G_J'],
                                    self.tube_parameters_high['x_curv']])
        parameter_space = gym.spaces.Box(low=np.tile(parameters_low, self.num_tubes),
                                         high=np.tile(parameters_high, self.num_tubes), dtype='float32')
        return parameter_space

    def sample_tube_parameters(self, num_discrete):
        # Constraints:
        # L >= L_c
        # d_i < d_o
        # L_1 >= L_2 >= L_3
        # di_1 >= di_2 >= di_3
        # do_1 >= do_2 >= do_3
        # xcurve_1 >= xcurve_2 >= xcurve_3
        tube_params = {}
        tube_parameters = []
        # Define sample space for each parameter
        E_I_sample_space = np.linspace(self.tube_parameters_low['E_I'], self.tube_parameters_low['E_I'], num_discrete)
        G_J_sample_space = np.linspace(self.tube_parameters_low['G_J'], self.tube_parameters_low['G_J'], num_discrete)
        L_sample_space = np.linspace(self.tube_parameters_low['L'], self.tube_parameters_high['L'], num_discrete)
        L_c_sample_space = np.linspace(self.tube_parameters_low['L_c'], self.tube_parameters_high['L_c'], num_discrete)
        d_i_sample_space = np.linspace(self.tube_parameters_low['d_i'], self.tube_parameters_high['d_i'], num_discrete)
        x_curv_sample_space = np.linspace(self.tube_parameters_low['x_curv'], self.tube_parameters_high['x_curv'],
                                          num_discrete)

        diameter_diff = 0.4e-3
        tube_sep = 0.1e-3
        tube_params['L'] = np.random.choice(L_sample_space)
        # Sample an L_c smaller than L
        tube_params['L_c'] = np.random.choice(L_c_sample_space[L_c_sample_space <= tube_params['L']])
        tube_params['d_i'] = np.random.choice(d_i_sample_space)
        tube_params['d_o'] = tube_params['d_i'] + diameter_diff
        tube_params['E_I'] = np.random.choice(E_I_sample_space)
        tube_params['G_J'] = np.random.choice(G_J_sample_space)
        tube_params['x_curv'] = np.random.choice(x_curv_sample_space)
        # Append as tube 0 parameters
        tube_parameters.append(tube_params)
        # iterate through tubes starting at tube 1
        #TODO: Diameter differences
        for i in range(1, self.num_tubes):
            tube_params = {}
            tube_params['L'] = np.random.choice(L_sample_space[L_sample_space <= tube_parameters[i - 1]['L']])
            tube_params['L_c'] = np.random.choice(L_c_sample_space[L_c_sample_space <= tube_params['L']])
            tube_params['d_i'] = tube_parameters[i-1]['d_o'] + tube_sep
            tube_params['d_o'] = tube_params['d_i'] + diameter_diff
            tube_params['E_I'] = tube_parameters[0]['E_I']
            tube_params['G_J'] = tube_parameters[0]['G_J']
            tube_params['x_curv'] = np.random.choice(
                x_curv_sample_space[x_curv_sample_space <= tube_parameters[i - 1]['x_curv']])
            tube_parameters.append(tube_params)
        return tube_parameters

    def get_observation_space(self):
        initial_tol = self.goal_tolerance_parameters['initial_tol']
        final_tol = self.goal_tolerance_parameters['final_tol']
        parameter_space = self.get_parameter_space()
        obs_space_low = np.concatenate(
            (np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -2 * 0.5, -2 * 0.5, -1.0, final_tol]),
             parameter_space.low))
        obs_space_high = np.concatenate(
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2 * 0.25, 2 * 0.25, 0.5, initial_tol]), parameter_space.high))
        observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=np.array([-0.5, -0.5, -1.0]), high=np.array([0.5, 0.5, 1.0]),
                                        dtype=float),
            achieved_goal=gym.spaces.Box(low=np.array([-0.5, -0.5, -1.0]), high=np.array([0.5, 0.5, 1.0]),
                                         dtype=float),
            observation=gym.spaces.Box(
                low=obs_space_low.flatten(),
                high=obs_space_high.flatten(),
                dtype=float)
        ))
        return observation_space

    def rep2joint(self, rep):
        rep = [rep[i:i + 3] for i in range(0, len(rep), 3)]
        beta = np.empty(self.num_tubes)
        alpha = np.empty(self.num_tubes)
        for tube in range(0, self.num_tubes):
            joint = self.single_trig2joint(rep[tube])
            alpha[tube] = joint[0]
            beta[tube] = joint[1]
        return np.concatenate((beta, alpha))

    def joint2rep(self, joint):
        rep = np.array([])
        betas = joint[:self.num_tubes]
        alphas = joint[self.num_tubes:]
        for beta, alpha in zip(betas, alphas):
            trig = self.single_joint2trig(np.array([beta, alpha]))
            rep = np.append(rep, trig)
        return rep

    # Single conversion from a joint to trig representation
    @staticmethod
    def single_joint2trig(joint):
        return np.array([np.cos(joint[1]),
                         np.sin(joint[1]),
                         joint[0]])

    # Single conversion from a trig representation to joint
    @staticmethod
    def single_trig2joint(trig):
        return np.array([np.arctan2(trig[1], trig[0]), trig[2]])
