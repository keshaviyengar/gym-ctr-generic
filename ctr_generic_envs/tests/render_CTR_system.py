import gym
import numpy as np
import ctr_generic_envs
from ctr_generic_envs.envs import ExactModel

from stable_baselines.her.utils import HERGoalEnvWrapper

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

# Given joint values and system, this script will plot the robot shape


if __name__ == '__main__':
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {}
    ctr_env =  HERGoalEnvWrapper(gym.make(env_id, **env_kwargs)).env.env
    num_systems = 60
    ctr_kine_model = ExactModel()

    fig = plt.figure()
    ax = plt.axes()
    q_list = [np.array([0,0,0,np.pi/2,np.pi/2,np.pi/2]),
         np.array([0, 0, 0, -np.pi / 2, np.pi / 2, np.pi / 2]),
         np.array([0, 0, 0, np.pi / 2, -np.pi / 2, np.pi / 2]),
         np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, np.pi / 2])]
    for system in range(0, num_systems):
        _ = ctr_env.reset()
        tube_params = ctr_env.tube_params
        labelled = True
        for q in q_list:
            ctr_kine_model.set_tube_parameters(tube_params)
            ee_pos = ctr_kine_model.forward_kinematics(q)
            r1, r2, r3 = ctr_kine_model.get_rs()
            if not labelled:
                labelled = True
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system*4), label='System ' + str(system))
            else:
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.twilight(system*2), alpha=0.5)
            ax.plot(r2[:,0] * 1000, r2[:,2] * 1000, linewidth=6.0, c=plt.cm.twilight(system*2 + 2), alpha=0.5)
            ax.plot(r3[:,0] * 1000, r3[:,2] * 1000, linewidth=8.0, c=plt.cm.twilight(system*2 + 4), alpha=0.5)
            ax.set_xlabel('$x$ (mm)')
            ax.set_ylabel('$z$ (mm)')
            ax.set_xlim([0, 350])
            ax.set_ylim([0, 500])
            ax.set_aspect('equal')
    ax.legend(loc='best')
    plt.grid()
    plt.show()
