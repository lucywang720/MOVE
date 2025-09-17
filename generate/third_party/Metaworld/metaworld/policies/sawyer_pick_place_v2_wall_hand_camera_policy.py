import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move, assert_fully_parsed

class SawyerPickPlaceWallHandCameraV2Policy(Policy):
    reset_grip = False  # 来自手部摄像头策略的属性

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        """结合两个策略的观测解析，适应手部摄像头环境"""
        # print(obs.shape)
        return {
            'hand_pos': obs[:3],
            'gripper_distance_apart': obs[3],  # 来自手部摄像头策略
            'puck_pos': obs[4:7],
            # 'unused_1': obs[7],  # 来自原始墙策略
            'puck_rot': obs[7:11],  # 来自手部摄像头策略
            'goal_pos': obs[-3:],
            'unused_info_curr_obs': obs[11:18],
            # 'unused_info': obs[12:-3],  # 合并未使用信息
            '_prev_obs':obs[18:36]  # 来自手部摄像头策略
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(
            o_d['hand_pos'],
            to_xyz=self._desired_pos(o_d),
            p=10.0
        )
        
        action['grab_effort'] = self._grab_effort(o_d) if not self.reset_grip else 0
        
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        """结合墙避障和手部摄像头抓取逻辑的位置控制"""
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([-0.005, 0, 0])
        pos_goal = o_d['goal_pos']
        gripper_separation = o_d['gripper_distance_apart']

        # 第一阶段：移动到puck上方
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02: # 0.015 in wall
            return pos_puck + np.array([0., 0., 0.1])
        
        # 第二阶段：下降到puck位置
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < (0.02 + o_d['puck_pos'][-1]): # 0.03 in wall
            return pos_puck + np.array([0., 0., 0.03])
        
        # 第三阶段：等待夹爪闭合
        elif gripper_separation > 0.73:
            return pos_curr
        
        # 第四阶段：向目标移动（加入墙避障逻辑）
        else:
            # 如果机械臂在墙区域内且高度低于0.25，先上移
            if (-0.15 <= pos_curr[0] <= 0.35 and 
                0.60 <= pos_curr[1] <= 0.80 and 
                pos_curr[2] < 0.25):
                return pos_curr + np.array([0, 0, 0.5])
            
            # 如果在墙区域内且高度低于0.35，保持高度移动
            elif (-0.15 <= pos_curr[0] <= 0.35 and 
                  0.60 <= pos_curr[1] <= 0.80 and 
                  pos_curr[2] < 0.35):
                return np.array([pos_goal[0], pos_goal[1], pos_curr[2]])
            
            # 如果与目标高度差较大，先调整高度
            elif abs(pos_curr[2] - pos_goal[2]) > 0.04:
                return np.array([pos_curr[0], pos_curr[1], pos_goal[2]])
            
            # 直接移向目标
            else:
                return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        """结合两个策略的抓取逻辑"""
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']
        
        # 当手和puck足够接近时闭合夹爪
        if np.linalg.norm(pos_curr - pos_puck) < 0.07:
            return 1.0
        else:
            return 0.0