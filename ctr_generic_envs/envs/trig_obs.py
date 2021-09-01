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


class TrigObs(object):
    def __init__(self, systems, goal_tolerance_parameters, noise_parameters, initial_q, relative_q, ext_tol):
        self.systems = systems
        self.tube_lengths = list()
        self.num_tubes = len(self.systems[0])
        for system in self.systems:
            tube_length = list()
            for tube in system:
                tube_length.append(tube.L)
            self.tube_lengths.append(tube_length)

        self.goal_tolerance_parameters = goal_tolerance_parameters
        self.noise_parameters = noise_parameters
        extension_std_noise = np.full(self.num_tubes, noise_parameters['extension_std'])
        rotation_std_noise = np.full(self.num_tubes, noise_parameters['rotation_std'])
        self.q_std_noise = np.concatenate((extension_std_noise, rotation_std_noise))
        self.tracking_std_noise = np.full(3, noise_parameters['tracking_std'])
        # Keep q as absolute joint positions, convert to relative as needed and store as absolute
        self.q = initial_q
        self.relative_q = relative_q

        # Q space, create per system
        self.q_spaces = list()
        for tube_betas in self.tube_lengths:
            self.q_spaces.append(gym.spaces.Box(low=np.concatenate((-np.array(tube_betas) + ext_tol,
                                                                   np.full(self.num_tubes, -np.inf))),
                                                high=np.concatenate((np.full(self.num_tubes, 0),
                                                                    np.full(self.num_tubes, np.inf)))
                                                ))
        # desired, achieved goal space
        self.observation_space = self.get_observation_space()

        self.goal_dim = 3

    def set_action(self, action, system_idx):
        self.q = np.clip(self.q + action, self.q_spaces[system_idx].low, self.q_spaces[system_idx].high)
        q_betas = self.q[:self.num_tubes]
        q_alphas = self.q[self.num_tubes:]
        for i in range(1, self.num_tubes):
            # Remember ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            q_betas[i - 1] = min(q_betas[i - 1], q_betas[i])
            q_betas[i - 1] = max(q_betas[i - 1],
                                 self.tube_lengths[system_idx][i] - self.tube_lengths[system_idx][i - 1] + q_betas[i])

        self.q = np.concatenate((q_betas, q_alphas))

    def sample_goal(self, system_idx):
        # while loop to get constrained points, maybe switch this for a workspace later on
        sample_counter = 0
        while True:
            q_goal_sample = self.q_spaces[system_idx]
            q_goal_sample.low[3:] = np.full(self.num_tubes, -2*np.pi)
            q_goal_sample.high[3:] = np.full(self.num_tubes, 2*np.pi)
            q_sample = q_goal_sample.sample()
            betas = q_sample[0:self.num_tubes]
            alphas = q_sample[self.num_tubes:]
            # Apply constraints
            valid_joint = []
            for i in range(1, self.num_tubes):
                valid_joint.append((betas[i - 1] <= betas[i]) and (
                        betas[i - 1] + self.tube_lengths[system_idx][i - 1] >= self.tube_lengths[system_idx][i] + betas[i]))
                # print(self.num_tubes)
                # print("q_sample: ", q_sample)
                # print("B", i - 1, " <= ", "B", i, " : ", q_sample[i - 1], " <= ", q_sample[i])
                # print("B", i - 1, " + L", i - 1, " <= ", "B", i, " + L", i, " : ",
                #       q_sample[i - 1] + self.tube_lengths[i - 1], " >= ", q_sample[i] + self.tube_lengths[i])
                # print("valid joint: ", valid_joint)
                # print("")
            sample_counter += 1
            if all(valid_joint):
                break
            if sample_counter > 1000:
                print("Stuck sampling goals...")
        q_constrain = np.concatenate((betas, alphas))
        return q_constrain

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance, system_idx):
        # Add noise to q, rotation and extension (encoder noise)
        noisy_q = np.random.normal(self.q, self.q_std_noise)
        # Add noise to achieved goal (tracker noise)
        noisy_achieved_goal = np.random.normal(achieved_goal, self.tracking_std_noise)
        # Relative joint representation
        if self.relative_q:
            rel_q = self.qabs2rel(noisy_q)
            rep = self.joint2rep(rel_q)
        else:
            rep = self.joint2rep(noisy_q)
        self.obs = {
            'desired_goal': (desired_goal).astype(np.float32),
            'achieved_goal': (noisy_achieved_goal).astype(np.float32),
            'observation': (np.concatenate(
                (rep, desired_goal - noisy_achieved_goal, np.array([goal_tolerance]))).astype(np.float32)
            )
        }
        #np.set_printoptions(precision=3)
        #if not self.observation_space["desired_goal"].contains(desired_goal):
        #    print("desired goal not in space.")
        #if not self.observation_space["achieved_goal"].contains(achieved_goal):
        #    print("achieved goal not in space.")
        #if not self.observation_space["observation"].contains(self.obs["observation"]):
        #    if not self.get_rep_space().contains(rep):
        #        if np.argwhere(rep < self.get_rep_space().low).size != 0:
        #            print("rep_val: ", rep[np.argwhere(rep < self.get_rep_space().low)])
        #            print("rep_low: ", self.get_rep_space().low)
        #        if np.argwhere(rep > self.get_rep_space().high).size != 0:
        #            print("rep_val: ", rep[np.argwhere(rep > self.get_rep_space().high)])
        #            print("rep_high: ", self.get_rep_space().high)
        #        print("rep: ", rep)
        #    else:
        #        print("goal error or tolerance out of bounds.")
        #        print("low: ", self.observation_space["observation"].low[9:])
        #        print("high: ", self.observation_space["observation"].high[9:])
        #        print("error and tol: ", np.concatenate((desired_goal - noisy_achieved_goal, np.array([goal_tolerance]))))
        return self.obs

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

    # TODO: Get min lengths
    def get_rep_space(self):
        rep_low = np.array([])
        rep_high = np.array([])
        # TODO: zero tol needs to be included in model and base class
        zero_tol = 1e-4
        max_tube_lengths = np.amax(np.array(self.tube_lengths), axis=0)
        for tube_length in max_tube_lengths:
            rep_low = np.append(rep_low, [-1, -1, -2*tube_length + zero_tol])
            rep_high = np.append(rep_high, [1, 1, 2*tube_length])
        rep_space = gym.spaces.Box(low=rep_low, high=rep_high, dtype="float32")
        return rep_space

    def get_observation_space(self):
        initial_tol = self.goal_tolerance_parameters['initial_tol']
        final_tol = self.goal_tolerance_parameters['final_tol']
        rep_space = self.get_rep_space()

        # TODO: re-add the system-idx
        obs_space_low = np.concatenate(
            (rep_space.low, np.array([-0.5, -0.5, -0.5, final_tol - 1e-4])))
        obs_space_high = np.concatenate(
            (rep_space.high, np.array([0.5, 0.5, 0.5, initial_tol + 1e-4])))
        observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=np.array([-0.5, -0.5, 0]), high=np.array([0.5, 0.5, 0.5]),
                                        dtype="float32"),
            achieved_goal=gym.spaces.Box(low=np.array([-0.5, -0.5, 0]), high=np.array([0.5, 0.5, 0.5]),
                                         dtype="float32"),
            observation=gym.spaces.Box(
                low=obs_space_low,
                high=obs_space_high,
                dtype="float32")
        ))
        self.obs_dim = obs_space_low.size
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
