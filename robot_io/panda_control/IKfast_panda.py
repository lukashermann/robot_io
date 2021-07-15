from ikfast_franka_panda import get_dof, get_fk, get_free_dof, get_ik
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R


class IKfast:
    def __init__(
        self,
        rp,
        joint_limits,
        # ll_real=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
        # ul_real=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
        weights=(1, 1, 1, 1, 1, 1, 1),
        num_angles=50,
    ):
        self.ll = joint_limits['lower']
        self.ul = joint_limits['upper']
        self.rp = rp
        self.num_dof = len(self.ll)
        self.weights = weights
        self.num_angles = num_angles

    def filter_solutions(self, sol):
        test_sol = np.ones(self.num_dof) * 9999.0
        for i in range(self.num_dof):
            for add_ang in [-2.0 * np.pi, 0, 2.0 * np.pi]:
                test_ang = sol[i] + add_ang
                if self.ul[i] >= test_ang >= self.ll[i]:
                    test_sol[i] = test_ang
        if np.all(test_sol != 9999.0):
            return test_sol
        return None

    def take_closest_sol(self, sols, last_q, weights):
        best_sol_ind = np.argmin(np.sum((weights * (sols - np.array(last_q))) ** 2, 1))
        return sols[best_sol_ind]

    def inverse_kinematics(self, target_pos, target_orn):

        sols = []
        feasible_sols = []
        for q_6 in np.linspace(self.ll[-1], self.ul[-1], self.num_angles):
            sols += get_ik(target_pos, target_orn, [q_6])
        for sol in sols:
            sol = self.filter_solutions(sol)
            if sol is not None:
                feasible_sols.append(sol)
        if len(feasible_sols) < 1:
            return False, None
        best_sol = self.take_closest_sol(feasible_sols, self.rp[:7], self.weights)
        return True, best_sol
