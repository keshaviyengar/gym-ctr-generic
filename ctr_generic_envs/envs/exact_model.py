import numpy as np
from math import pi, pow
from scipy.integrate import odeint, solve_ivp

from ctr_generic_envs.envs.CTR_Python import Segment
from ctr_generic_envs.envs.CTR_Python import CTR_Model
from ctr_generic_envs.envs.CTR_Python import Tube

from mpi4py import MPI

from copy import deepcopy


def sample_parameters(tube_parameters, randomization):
    L = randomize_value(tube_parameters.L, 0)
    L_c = randomize_value(tube_parameters.L_c, 0)
    diameter_inner = randomize_value(tube_parameters.diameter_inner, randomization)
    diameter_outer = randomize_value(tube_parameters.diameter_outer, randomization)
    stiffness = randomize_value(tube_parameters.E, randomization)
    torsional_stiffness = randomize_value(tube_parameters.G, randomization)
    x_curvature = randomize_value(tube_parameters.U_x, randomization)
    y_curvature = randomize_value(tube_parameters.U_y, 0)

    new_parameters = Tube(L, L_c, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature,
                          y_curvature)
    return new_parameters


def randomize_value(value, randomization):
    sampled_value = np.random.uniform(value - value * randomization, value + value * randomization)
    return sampled_value


class ExactModel(object):
    def __init__(self):
        self.num_tubes = 3
        self.r = []
        self.r1 = []
        self.r2 = []
        self.r3 = []
        self.r_transforms = []

    # Randomize parameters by a percentage
    #def randomize_parameters(self, randomization):
    #    for i in range(len(self.curr_systems)):
    #        self.curr_systems[i][0] = sample_parameters(self.systems[i][0], randomization)
    #        self.curr_systems[i][1] = sample_parameters(self.systems[i][1], randomization)
    #        self.curr_systems[i][2] = sample_parameters(self.systems[i][2], randomization)

    def set_tube_parameters(self, tube_parameters):
        self.t0 = Tube(tube_parameters[0]['L'], tube_parameters[0]['L_c'], tube_parameters[0]['d_i'],
                       tube_parameters[0]['d_o'], tube_parameters[0]['E_I'], tube_parameters[0]['G_J'],
                       tube_parameters[0]['x_curv'], 0)
        self.t1 = Tube(tube_parameters[1]['L'], tube_parameters[1]['L_c'], tube_parameters[1]['d_i'],
                       tube_parameters[1]['d_o'], tube_parameters[1]['E_I'], tube_parameters[1]['G_J'],
                       tube_parameters[1]['x_curv'], 0)
        self.t2 = Tube(tube_parameters[2]['L'], tube_parameters[2]['L_c'], tube_parameters[2]['d_i'],
                       tube_parameters[2]['d_o'], tube_parameters[2]['E_I'], tube_parameters[2]['G_J'],
                       tube_parameters[2]['x_curv'], 0)

    def forward_kinematics(self, q, **kwargs):
        """
        q_0 = np.array([0, 0, 0, 0, 0, 0])
        # initial twist (for ivp solver)
        uz_0 = np.array([0.0, 0.0, 0.0])
        u1_xy_0 = np.array([[0.0], [0.0]])
        # force on robot tip along x, y, and z direction
        f = np.array([0, 0, 0]).reshape(3, 1)

        # Use this command if you wish to use initial value problem (ivp) solver (less accurate but faster)
        CTR = CTR_Model(self.systems[0][0], self.systems[0][1], self.systems[0][2], f, q, q_0, 0.01, 1)
        cost = CTR.minimize(np.concatenate((u1_xy_0, uz_0), axis=None))
        return CTR.r[-1]
        """
        # position of tubes' base from template (i.e., s=0)
        q_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        beta = q[0:3] + q_0[0:3]

        segment = Segment(self.t0, self.t1, self.t2, beta)

        r_0_ = np.array([0, 0, 0]).reshape(3, 1)
        alpha_1_0 = q[3] + q_0[3]
        R_0_ = np.array(
            [[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]]) \
            .reshape(9, 1)
        alpha_0_ = q[3:].reshape(3, 1) + q_0[3:].reshape(3, 1)

        # initial twist
        uz_0_ = np.array([0, 0, 0])
        self.r, U_z, tip = self.ctr_model(uz_0_, alpha_0_, r_0_, R_0_, segment, beta)
        try:
            self.r1 = self.r[tip[1]:tip[0] + 1]
            self.r2 = self.r[tip[2]:tip[1] + 1]
            self.r3 = self.r[:tip[2] + 1]
            assert not np.any(np.isnan(self.r))
            return self.r[-1]
        except TypeError:
            return np.zeros(3)

    def get_r(self):
        return self.r

    def get_rs(self):
        return self.r1, self.r2, self.r3

    def get_r_transforms(self):
        return self.r_transforms

    # ode equation
    def ode_eq(self, s, y, ux_0, uy_0, ei, gj):
        dydt = np.empty([18, 1])
        ux = np.empty([3, 1])
        uy = np.empty([3, 1])
        for i in range(0, 3):
            ux[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                    (ei[0] * ux_0[0] * np.cos(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.sin(y[3 + i] - y[3 + 0]) +
                     ei[1] * ux_0[1] * np.cos(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.sin(y[3 + i] - y[3 + 1]) +
                     ei[2] * ux_0[2] * np.cos(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.sin(y[3 + i] - y[3 + 2]))
            uy[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                    (-ei[0] * ux_0[0] * np.sin(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.cos(y[3 + i] - y[3 + 0]) +
                     -ei[1] * ux_0[1] * np.sin(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.cos(y[3 + i] - y[3 + 1]) +
                     -ei[2] * ux_0[2] * np.sin(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.cos(y[3 + i] - y[3 + 2]))

        for j in range(0, 3):
            if ei[j] == 0:
                dydt[j] = 0  # ui_z
                dydt[3 + j] = 0  # alpha_i
            else:
                dydt[j] = ((ei[j]) / (gj[j])) * (ux[j] * uy_0[j] - uy[j] * ux_0[j])  # ui_z
                dydt[3 + j] = y[j]  # alpha_i

        e3 = np.array([0, 0, 1]).reshape(3, 1)
        uz = y[0:3]
        R = np.array(y[9:]).reshape(3, 3)
        u_hat = np.array([(0, - uz[0], uy[0]), (uz[0], 0, -ux[0]), (-uy[0], ux[0], 0)])
        dr = np.dot(R, e3)
        dR = np.dot(R, u_hat).ravel()

        dydt[6] = dr[0]
        dydt[7] = dr[1]
        dydt[8] = dr[2]

        for k in range(3, 12):
            dydt[6 + k] = dR[k - 3]
        return dydt.ravel()

    # CTR model
    def ctr_model(self, uz_0, alpha_0, r_0, R_0, segmentation, beta):
        tube1 = self.t0
        tube2 = self.t1
        tube3 = self.t2
        Length = np.empty(0)
        r = np.empty((0, 3))
        u_z = np.empty((0, 3))
        alpha = np.empty((0, 3))
        span = np.append([0], segmentation.S)
        for seg in range(0, len(segmentation.S)):
            # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
            y_0 = np.vstack((uz_0.reshape(3, 1), alpha_0, r_0, R_0)).ravel()
            s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
            # s = odeint(self.ode_eq, y_0, s_span, args=(
            #    segmentation.U_x[:, seg], segmentation.U_y[:, seg], segmentation.EI[:, seg], segmentation.GJ[:, seg]),
            #           tfirst=True)
            if np.all(np.diff(s_span) < 0):
                print("s_span not sorted correctly. Resorting...")
                print("linespace: ", s_span[seg], s_span[seg + 1] - 1e-6)
                s_span = np.sort(s_span)
            sol = solve_ivp(fun=lambda s, y: self.ode_eq(s, y, segmentation.U_x[:, seg], segmentation.U_y[:, seg],
                                                         segmentation.EI[:, seg], segmentation.GJ[:, seg]),
                            t_span=(min(s_span), max(s_span)), y0=y_0, t_eval=s_span)
            if sol.status == -1:
                print(sol.message)
            s = np.transpose(sol.y)
            Length = np.append(Length, s_span)
            u_z = np.vstack((u_z, s[:, (0, 1, 2)]))
            alpha = np.vstack((alpha, s[:, (3, 4, 5)]))
            r = np.vstack((r, s[:, (6, 7, 8)]))

            # new boundary conditions for next segment
            r_0 = r[-1, :].reshape(3, 1)
            R_0 = np.array(s[-1, 9:]).reshape(9, 1)
            uz_0 = u_z[-1, :].reshape(3, 1)
            alpha_0 = alpha[-1, :].reshape(3, 1)

        d_tip = np.array([tube1.L, tube2.L, tube3.L]) + beta
        u_z_end = np.array([0.0, 0.0, 0.0])
        tip_pos = np.array([0, 0, 0])
        for k in range(0, 3):
            try:
                b = np.argmax(Length >= d_tip[k] - 1e-3)  # Find where tube curve starts
                u_z_end[k] = u_z[b, k]
                tip_pos[k] = b
            except ValueError:
                r = np.zeros(3)
                u_z_end = np.zeros(3)
                tip_pos = np.zeros(3)
                return r, u_z_end, tip_pos

        return r, u_z_end, tip_pos

    """
    # q[0:2] are extension values and q[3:5] rotation values from the base
    # in general q[..] = [betas.., alphas]
    # q0 are starting joint values
    # uz_0 is column vector of initial torsion about z axis
    # r_0 is base position
    # R_0 is initial base orientation
    def forward_kinematics(self, q, system_idx, **kwargs):
        if 'q_0' not in kwargs:
            kwargs['q_0'] = np.zeros(self.num_tubes * 2)
        if 'r_0' not in kwargs:
            kwargs['r_0'] = np.array([0, 0, 0]).reshape(3, 1)
        if 'uz_0' not in kwargs:
            # initial twist
            kwargs['uz_0'] = np.zeros(self.num_tubes)

        # Set tubes to selected system
        tubes = self.systems[system_idx]

        # Separate q_beta and q_alpha
        q_beta = np.array(q[:self.num_tubes])
        q_alpha = np.array(q[self.num_tubes:])

        q_0 = kwargs['q_0']
        q_0_beta = q_0[:self.num_tubes]
        q_0_alpha = q_0[self.num_tubes:]
        uz_0 = kwargs['uz_0']
        r_0 = kwargs['r_0']
        alpha_1_0 = q_alpha[0] + q_0_alpha[0]
        R_0 = np.array(
            [[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]]) \
            .reshape(9, 1)
        alpha_0 = q_alpha.reshape(self.num_tubes, 1) + q_0_alpha.reshape(self.num_tubes, 1)

        # position of tubes' base from template (i.e., s=0)
        beta = q_beta + q_0_beta
        segments = SegmentRobot(beta, tubes)

        Length = np.empty(0)
        r = np.empty((0, 3))
        r_transforms = np.empty((len(segments.S), 4, 4))
        u_z = np.empty((0, self.num_tubes))
        alpha = np.empty((0, self.num_tubes))
        span = np.append([0], segments.S)
        for seg in range(0, len(segments.S)):
            # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
            y_0 = np.vstack((uz_0.reshape(self.num_tubes, 1), alpha_0, r_0, R_0)).ravel()
            s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
            s = odeint(self.ode_eq, y_0, s_span, args=(segments.U_x[:, seg], segments.U_y[:, seg],
                                                       segments.EI[:, seg], segments.GJ))
            Length = np.append(Length, s_span)
            u_z = np.vstack((u_z, s[:, 0:self.num_tubes]))
            alpha = np.vstack((alpha, s[:, self.num_tubes:2*self.num_tubes]))
            r = np.vstack((r, s[:, 2*self.num_tubes:2*self.num_tubes+3]))

            # Start with identity
            transform = np.identity(4)
            # Add rotation
            transform[:3, :3] = np.array(s[-1, -9:]).reshape(3, 3)
            # Add position
            transform[0:3, 3] = np.array(s[-1, 2*self.num_tubes:2*self.num_tubes+3]).reshape(3)
            r_transforms[seg, :, :] = transform

            # new boundary conditions for next segment
            r_0 = r[-1, :].reshape(3, 1)
            R_0 = np.array(s[-1, -9:]).reshape(9, 1)
            uz_0 = u_z[-1, :].reshape(self.num_tubes, 1)
            alpha_0 = alpha[-1, :].reshape(self.num_tubes, 1)

        tube_lengths = [i.L for i in tubes]
        d_tip = tube_lengths + beta
        u_z_end = np.zeros(self.num_tubes)
        tip_pos = np.zeros(self.num_tubes)
        for k in range(0, self.num_tubes):
            try:
                b = np.argmax(Length >= d_tip[k] - 1e-3)  # Find where tube curve starts
            except ValueError:
                print("Error in b calculation")
                b = 0
            try:
                u_z_end[k] = u_z[b, k]
                tip_pos[k] = b
            except IndexError:
                print("Index for u_z is zero b is NaN")
                u_z_end[k] = 0
                tip_pos[k] = 0
        self.r = r
        self.r_transforms = r_transforms
        if len(r) == 0:
            print("r is zero. Relevant debug info:")
            print("q: ", q)
        r1 = r
        _, idx = min((val, idx) for (idx, val) in enumerate(abs(Length - segments.d_tip[len(segments.d_tip) - 2])))
        r2 = r[:idx, :]
        _, idx = min((val, idx) for (idx, val) in enumerate(abs(Length - segments.d_tip[len(segments.d_tip) - 1])))
        r3 = r[:idx, :]
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        return r[-1]
    """
