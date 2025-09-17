import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from scipy.spatial.transform import Rotation
import math

def compute_ee_quat(drawer_quat, offset_rotation=None):
    """
    根据抽屉四元数计算机械臂末端目标四元数
    :param drawer_quat: 抽屉四元数 (w, x, y, z)
    :param offset_rotation: 可选附加旋转（欧拉角或四元数）
    :return: 机械臂末端四元数 (w, x, y, z)
    """
    # 转换抽屉四元数为Scipy格式 (x, y, z, w)
    drawer_rot = Rotation.from_quat(np.roll(drawer_quat, -1))
    
    # 定义末端相对于抽屉的期望旋转（默认Z轴对齐）
    if offset_rotation is None:
        # 示例：末端Z轴与抽屉Z轴对齐，X轴旋转-90度（适应抓握）
        offset_rot = Rotation.from_euler('y', 90, degrees=True)
    else:
        offset_rot = Rotation.from_euler(offset_rotation[0], offset_rotation[1], degrees=True)
    
    # 组合旋转：全局抽屉方向 + 局部调整
    target_rot = drawer_rot * offset_rot
    target_quat = target_rot.as_quat()  # (x, y, z, w)
    
    # 转换回MuJoCo格式 (w, x, y, z)
    return np.roll(target_quat, 1)



# class SawyerDrawerOpenV2Policy(Policy):

#     @staticmethod
#     @assert_fully_parsed
#     def _parse_obs(obs):
#         return {
#             'hand_pos': obs[:3],
#             'gripper': obs[3],
#             'drwr_pos': obs[4:7],

#             'unused_info': obs[7:],
#         }

#     def get_action(self, obs):
#         o_d = self._parse_obs(obs)

#         action = Action({
#             'delta_pos': np.arange(3),
#             'grab_effort': 3
#         })

#         # NOTE this policy looks different from the others because it must
#         # modify its p constant part-way through the task
#         pos_curr = o_d['hand_pos']
#         pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

#         # align end effector's Z axis with drawer handle's Z axis
#         if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
#             to_pos = pos_drwr + np.array([0., 0., 0.3])
#             action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
#         # drop down to touch drawer handle
#         elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
#             to_pos = pos_drwr
#             action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
#         # push toward a point just behind the drawer handle
#         # also increase p value to apply more force
#         else:
#             to_pos = pos_drwr + np.array([0., -0.06, 0.])
#             action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=50.)

#         # keep gripper open
#         action['grab_effort'] = -1.

#         return action.array


class SawyerDrawerOpenRotateV2Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'drwr_pos': obs[4:7],
            'drwr_quat': obs[7:11],  # 四元数位置需根据实际obs结构修改
            'v_move': obs[11],
            'd_move': obs[12:15],
            'unused_info': obs[15:],
        }

    # def get_action(self, obs):
    #     print(obs)
    #     o_d = self._parse_obs(obs)
    #     action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        
    #     pos_curr = o_d['hand_pos']
    #     # pos_drwr = update_target_pos(o_d['drwr_quat'], o_d['drwr_pos']) + np.array([.0, .0, -.02])
    #     pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])
        
    #     # 计算推动方向
    #     rot_matrix = Rotation.from_quat(o_d['drwr_quat']).as_matrix()
    #     local_push_dir = np.array([0, 1, 0])  # 局部坐标系推动方向
    #     global_push_dir = rot_matrix.dot(local_push_dir) * 0.06
        
    #     # 对齐阶段逻辑保持不变
    #     if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
    #         to_pos = pos_drwr + np.array([0., 0., 0.3])
    #         action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
    #     elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
    #         to_pos = pos_drwr
    #         action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
    #     # 修改后的推动逻辑
    #     else:
    #         to_pos = pos_drwr + global_push_dir
    #         action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=50.)
        
    #     action['grab_effort'] = -1.
    #     print("action", action.array, o_d['drwr_quat'])
    #     print(np.concatenate([action.array, np.array(o_d['drwr_quat'])], axis=0))
    #     return action.array
    #     # return np.concatenate([action.array, np.array(o_d['drwr_quat'])], axis=0)



    def get_action(self, obs):
        # print(obs)
        o_d = self._parse_obs(obs)
        action = Action({'delta_pos': np.arange(3), 'grab_effort': 3})
        
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02]) + o_d['v_move']*o_d['d_move']
        # pos_drwr = np.array([o_d['drwr_pos'][0]+0.16*math.sin(2*math.asin(o_d['drwr_quat'][-1])), o_d['drwr_pos'][1]-0.16*math.cos(2*math.asin(o_d['drwr_quat'][-1]))+0.16, pos_drwr[2]])
        pos_drwr = np.array([o_d['drwr_pos'][0]+0.18*math.sin(2*math.asin(o_d['drwr_quat'][-1])), o_d['drwr_pos'][1]-0.18*math.cos(2*math.asin(o_d['drwr_quat'][-1]))+0.18, o_d['drwr_pos'][2]])


        # 计算推动方向
        # rot_matrix = Rotation.from_quat(compute_ee_quat(o_d['drwr_quat'], ['y', -90])).as_matrix()
        rot_matrix = Rotation.from_quat(compute_ee_quat(o_d['drwr_quat'], ['y', -90])).as_matrix()
        local_push_dir = np.array([0, 1, 0])  # 局部坐标系推动方向
        global_push_dir = rot_matrix.dot(local_push_dir) * 0.06
        
        # 对齐阶段逻辑保持不变
        if np.linalg.norm(pos_curr[:2] - pos_drwr[:2]) > 0.06:
            to_pos = pos_drwr + np.array([0., 0., 0.3])
            action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
        elif abs(pos_curr[2] - pos_drwr[2]) > 0.04:
            to_pos = pos_drwr
            action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=4.)
        else:
            to_pos = pos_drwr + global_push_dir
            # print("pos_drwr: ", pos_drwr, o_d['drwr_quat'])
            action['delta_pos'] = move(o_d['hand_pos'], to_pos, p=50.)
        
        action['grab_effort'] = -1.
        return np.concatenate([action.array, np.array(o_d['drwr_quat'])], axis=0)