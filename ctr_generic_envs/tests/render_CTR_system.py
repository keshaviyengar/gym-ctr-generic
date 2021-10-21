import gym
import numpy as np
import ctr_generic_envs
from ctr_generic_envs.envs import ExactModel

from stable_baselines.her.utils import HERGoalEnvWrapper

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# Given joint values and system, this script will plot the robot shape


if __name__ == '__main__':
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'num_systems': 4}
    ctr_env =  HERGoalEnvWrapper(gym.make(env_id, **env_kwargs)).env.env
    ctr_systems = ctr_env.systems
    ctr_kine_model = ExactModel(ctr_systems)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1,1,1])
    q = np.array([0,0,0,0,0,0])
    for system_idx in range(0, len(ctr_systems)):
        ee_pos = ctr_kine_model.forward_kinematics(q, system_idx)
        r1, r2, r3 = ctr_kine_model.get_rs()
        ax.plot3D(r1[:,0], r1[:,1], r1[:,2], linewidth=2.0)
        ax.plot3D(r2[:,0], r2[:,1], r2[:,2], linewidth=3.0)
        ax.plot3D(r3[:,0], r3[:,1], r3[:,2], linewidth=4.0)
    plt.show()
