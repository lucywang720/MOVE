import numpy as np
import math

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from scipy.spatial.transform import Rotation as R

def rotate_local(quat, O_global, P_local, switch_quat=False):
    # print(quat)
    quat = np.roll(quat, shift=-1) 
    if switch_quat:
        quat[:3] *= -1
    rotation = R.from_quat(quat)
    P_rotated = rotation.apply(P_local)
    # print(P_local, P_rotated, quat, P_rotated, rotation)
    # exit(0)
    P_world = O_global + P_rotated
    return P_world

# def transform_quat(quat):
#     # print(quat, math.acos(quat[0])/math.pi*180*2)
#     if quat[0] < math.sqrt(2)/2:
#         return np.array([quat[-1],0,0,-quat[0]])
#     else:
#         return quat


class SawyerAssemblyRotateAndBothV2Policy(Policy):

    grasping = False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'wrench_pos': obs[4:7],
            'peg_pos': obs[-3:],
            'obj_quat': obs[7:11],
            'unused_info': obs[11:-6],
            'delta_wrench': obs[29:32]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        # return np.concatenate([action.array, transform_quat(np.array(o_d['obj_quat']))], axis=0)
        return np.concatenate([action.array, np.array(o_d['obj_quat'])], axis=0)

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_peg = rotate_local(o_d['obj_quat'], o_d['peg_pos'], np.array([.12, .0, .14]))
        # print(pos_peg, o_d['peg_pos'])
        # exit(0)

        # pos_wrench = rotate_local(o_d['obj_quat'], o_d['unused_info'][6:9], np.array([.1, -0.0, 0]))
        pos_wrench = rotate_local(o_d['obj_quat'], o_d['wrench_pos']-o_d['delta_wrench'], np.array([.105, -0.0, 0.0]))
        # print("pos_wrench: ", pos_wrench, o_d['obj_quat'], o_d['wrench_pos'])
        # pos_wrench = o_d['wrench_pos']
        # print(pos_curr, pos_wrench)



        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0., 0., 0.1])
        # (For later) if lined up with peg, drop down on top of it
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
            return pos_peg + np.array([.0, .0, -.2])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench + np.array([0., 0., 0.03])
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # If XY error is greater than 0.02, place end effector above the peg
        else:
            return pos_peg

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = rotate_local(o_d['obj_quat'], o_d['wrench_pos']-o_d['delta_wrench'], np.array([.105, -0.0, 0.0]))

        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 or abs(pos_curr[2] - pos_wrench[2]) > 0.12:
            return 0.
        # Until hovering over peg, keep hold of wrench
        else:
            return 0.6
