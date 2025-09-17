import numpy as np
import math
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_assembly_peg_v2_all import random_z_rotation_quaternion
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_hand_camera_v2 import CameraController # use camera controller from pick place env
from metaworld.policies.sawyer_assembly_v2_all_policy import rotate_local

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ])
class SawyerHammerEnvV2(SawyerXYZEnv):
    """
    target: nail, goal in box
    object: hammer
    """
    HAMMER_HANDLE_LENGTH = 0.14

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.3, 0.4, 0.0)
        obj_high = (0.3, 0.5, 0.0)
        goal_low = (0.2399, .7399, 0.109)
        goal_high = (0.2401, .7401, 0.111)
        goal_low = (-0.24, .87, 0)
        goal_high = (0.24, .95, 0)
        obj_rot_low = (-np.pi/3)
        obj_rot_high = (np.pi/3)

        self.hand_low = hand_low
        self.hand_high = hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        self.goal_low = goal_low
        self.goal_high = goal_high
        self.retarget_step = 80   # like sawyer_assembly_peg_v2_all.py
        self.not_grasp_final = True
        self.obj_x = np.linspace(obj_low[0]+0.02, obj_high[0]-0.02, 5)
        self.obj_y = np.linspace(obj_low[1]+0.02, obj_high[1]-0.02, 4)


        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'hammer_init_pos': np.array([0, 0.5, 0.0]),
            'hand_init_pos': np.array([0, 0.4, 0.2]),
        }
        self.goal = self.init_config['hammer_init_pos']
        self.hammer_init_pos = self.init_config['hammer_init_pos']
        self.obj_init_pos = self.hammer_init_pos.copy()
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.nail_init_pos = None

        # self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
        # allow randomize both the object position and goal position
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low, obj_rot_low)),
            np.hstack((obj_high, goal_high, obj_rot_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        # set test configurations for hand, camera, hammer, nail
        self.obj_initial_cnt = 0
        # Initialize hand
        self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]
        # Initialize camera controller
        self.camera_controller = CameraController(self)
        self.theta_cam = [np.pi/21*i for i in range(1,21)]
        # Initialize hammer
        ...
        self.theta_hammer = np.linspace(-np.pi/3, np.pi/3, 10)
        # Initialize nail
        ...
        box_x = np.linspace(goal_low[0]+0.02, goal_high[0]-0.02,5)
        box_y = np.linspace(goal_low[1]+0.02, goal_high[1]-0.02,4)
        self.test_config = {
            'hammer_init_pos': np.array([[x,y,0] for x in self.obj_x for y in self.obj_y]),
            'hammer_init_rot': np.linspace(-np.pi/3, np.pi/3, 10),
            'hand_init_pos': np.array([[self.hand_x[i%10], self.hand_y[i%4], self.hand_z[i%5]] for i in range(20)]),
            'box_init_pos': np.array([[x,y,0] for x in box_x  for y in box_y])
        }


    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_hammer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('HammerHandle')

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('hammer').copy(),
            self.get_body_com('nail_link').copy()
        ))

    def _get_quat_objects(self):
        return np.hstack((
            self.sim.data.get_body_xquat('hammer'),
            # self.sim.model.body_quat[self.model.geom_name2id('HammerHandle')],
            self.sim.data.get_body_xquat('nail_link')
        ))

    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_hammer_quat(self, quat):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self, set_xyz=None):
        # print('random_init=', self.random_init)
        # self._reset_hand()

        # Set position of box & nail (these are not randomized)
        #  np.array([0.24, 0.85, 0.0]) # configured in sawyer_hammer.xml
        self.box_pos = self.test_config['box_init_pos'][self.obj_initial_cnt % 20]
        # Update _target_pos
        
        # reset hammer & camera (default)
        self.hammer_init_pos = np.array([
            self.obj_x[self.obj_initial_cnt % 5], 
            self.obj_y[self.obj_initial_cnt % 4], 
            0])
        self.hammer_init_rot = self.theta_hammer[self.obj_initial_cnt % 10]
        self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)

        self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])

        # Randomize init position for hammer & camera
        if self.random_init:
            tmp = self._get_state_rand_vec() #(hammer_pos, target, hammer_rot)
            while np.linalg.norm(tmp[:2] - tmp[3:5]) < 0.15 :  # too close to target
                tmp = self._get_state_rand_vec()
            # self._target_pos = tmp[3:]
            self.box_pos = tmp[3:6]
            self.hammer_init_pos = tmp[:3]
            self.hammer_init_rot = tmp[-1]
            self.camera_controller.randomize_camera(method='cylindrical')
            self.hand_init_pos = np.random.uniform(low=np.array([-0.4, 0.4, 0.1]), high=np.array([0.4, 0.6, 0.4]))
        else:
            self.obj_initial_cnt += 1    
        # self.hand_init_pos[2] = max(self.hand_init_pos[2], 0.1)
        self._reset_hand()
        self.init_left_pad = self.get_body_com('leftpad')
        self.init_right_pad = self.get_body_com('rightpad')

        # set hammer
        self.obj_init_pos = self.hammer_init_pos.copy()
        self._set_hammer_xyz(self.hammer_init_pos)
        self._set_hammer_quat(random_z_rotation_quaternion(self.hammer_init_rot/np.pi*180))
        # self._set_obj_xyz(self.obj_init_pos)
        # self.sim.model.body_quat[self.model.body_name2id('hammer')] = random_z_rotation_quaternion(self.hammer_init_rot/np.pi*180)
        
        # self.sim.model.body_quat[self.model.geom_name2id('HammerHandle')] = random_z_rotation_quaternion(self.hammer_init_rot/np.pi*180)
        self.sim.model.body_pos[self.model.body_name2id(
            'box'
        )] = self.box_pos
        self.sim.forward()

        #update state
        self._target_pos = self._get_site_pos('goal')
        self.nail_init_pos = self._get_site_pos('nailHead')
        self.hammer_init_quat = self.sim.data.get_body_xquat('hammer')
        self.hammer_init_pos = self.sim.data.get_body_xpos('hammer')
        
        # print(f"hammer_init\npos=\n{self.hammer_init_pos}\nquat=\n{self.hammer_init_rot}\n")
        return self._get_obs()

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [1, 0, 0, 0]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([1., 0., 0., 0.])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(hammer_head, target_pos):
        pos_error = target_pos - hammer_head

        a = 0.1  # Relative importance of just *trying* to lift the hammer
        b = 0.9  # Relative importance of hitting the nail
        lifted = hammer_head[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid='long_tail',
        )

        return in_place

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        hammer = obs[4:7]
        hammer_quat = obs[7:11]
        hammer_head = hammer + np.array([.16, .06, .0])
        # consider rotate
        hammer_head = rotate_local(hammer_quat, hammer, np.array([.16, .06, .0]))
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the hammer handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified hammer position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        hammer_threshed = hammer.copy()
        threshold = SawyerHammerEnvV2.HAMMER_HANDLE_LENGTH / 2.0
        if abs(hammer[0] - hand[0]) < threshold:
            hammer_threshed[0] = hand[0]

        reward_quat = SawyerHammerEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions, hammer_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )
        reward_in_place = SawyerHammerEnvV2._reward_pos(
            hammer_head,
            self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success. We check that reward is above a threshold
        # because this env's success metric could be hacked easily
        success = self.data.get_joint_qpos('NailSlideJoint') > 0.09
        if success and reward > 5.:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )

    # =====copy from sawyer_assembly_all_v2.py=====
    def set_new_target(self, v, d):
        # print("set nes target!")
        """
        possible functions:
            if v is set , move direction d
            if v is not set , randomly move  
        """
        if v is not None:
            # print(v,v*d[0],d[0])
            self.box_pos = np.clip(self.box_pos + v*d[0], self.goal_low, self.goal_high)
            # self._target_pos[-1] = self.sim.model.body_pos[self.model.body_name2id("tablelink")][-1]+0.1

            if self.box_pos[0] == self.goal_low[0] or self.box_pos[0] == self.goal_high[0]:
                d[0][0] = -d[0][0]
            if self.box_pos[1] == self.goal_low[1] or self.box_pos[1] == self.goal_high[1]:
                d[0][1] = -d[0][1]

            # peg_pos = self._target_pos - np.array([0., 0., 0.05])
            self.sim.model.body_pos[self.model.body_name2id('box')] = self.box_pos
            self.sim.forward()
            self._target_pos = self._get_site_pos('goal')
            # self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
            # self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos

            return d


    def set_new_obj(self, v, d):
        if v != 0:
            self.obj_init_pos = self.sim.data.get_body_xpos('hammer')
            self.obj_init_pos[:-1] = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)[:-1]
            # self.obj_init_pos[-1] = self.sim.model.body_pos[self.model.body_name2id("tablelink")][-1]+0.02
            self._set_obj_xyz(self.obj_init_pos)

            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]

            return d

    def rotate_obj(self, state=None, v=0, d=None):
        """

        """
        # print("time=",self.sim.get_state().time)
        if v != 0:
            # self._set_obj_xyz(self.obj_init_pos)
            quat0 = self.sim.data.get_body_xquat('hammer')
            dquat = random_z_rotation_quaternion(v*d*0.1) # fps=10
            self.hammer_init_quat = quat_mult(dquat, quat0)
            self._set_hammer_quat(self.hammer_init_quat)
            w, x, y, z = self.hammer_init_quat
            yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
            # drot = math.degrees(2*math.asin(self.sim.data.get_body_xquat('hammer')[-1]))
            # print(drot)
            # drot = math.degrees(2*math.asin(self.sim.model.body_quat[self.model.geom_name2id('HammerHandle')][-1]))
            # self.sim.model.body_quat[self.model.body_name2id('hammer')] = random_z_rotation_quaternion(drot+v*d)
            # self._set_hammer_quat(random_z_rotation_quaternion(drot+v*d))
            # self.sim.model.body_quat[self.model.geom_name2id('HammerHandle')] = random_z_rotation_quaternion(drot+v*d)
            
            # self.sim.forward()
            # SawyerNutAssemblyAllEnvV2.delta_xyz = self.data.site_xpos[self.model.site_name2id('RoundNut-8')] - self.data.site_xpos[self.model.site_name2id('RoundNut')]
            # theta = math.degrees(2*math.asin(self.sim.data.get_body_xquat('hammer')[-1]))
            # theta = 2*math.asin(self.sim.model.body_quat[self.model.geom_name2id('HammerHandle')][-1])/math.pi*180

            if yaw < -60 or yaw > 60:
                d = -d

            return d

        else:
            raise ValueError("v should not be 0, please set a valid v value")
            exit(0)   


    def set_rotate_and_new_obj(self, state=None, v_rotate=0, d_rotate=None, v_move=0, d_move=None):
        # print(v_rotate, v_move)
        if v_rotate != 0 and v_move != 0:
            d_move = self.set_new_obj(v=v_move, d=d_move)
            d_rotate = self.rotate_obj(v=v_rotate, d=d_rotate)
        elif v_rotate != 0:
            d_rotate = self.rotate_obj(v=v_rotate, d=d_rotate)
        elif v_move != 0:
            d_move = self.set_new_obj(v=v_move, d=d_move)
        else:
            raise ValueError("v_rotate and v_move should not be both 0, please set a valid v value")
            exit(0)  

        return d_rotate, d_move

# import numpy as np
# import math
# from gym.spaces import Box

# from metaworld.envs import reward_utils
# from metaworld.envs.asset_path_utils import full_v2_path_for
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
# from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_assembly_peg_v2_all import random_z_rotation_quaternion
# from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_hand_camera_v2 import CameraController # use camera controller from pick place env
# from metaworld.policies.sawyer_assembly_v2_all_policy import rotate_local


# class SawyerHammerEnvV2(SawyerXYZEnv):
#     """
#     target: nail, goal in box
#     object: hammer
#     """
#     HAMMER_HANDLE_LENGTH = 0.14

#     def __init__(self):
#         hand_low = (-0.5, 0.40, 0.05)
#         hand_high = (0.5, 1, 0.5)
#         obj_low = (-0.1, 0.4, 0.0)
#         obj_high = (0.1, 0.5, 0.0)
#         goal_low = (0.2399, .7399, 0.109)
#         goal_high = (0.2401, .7401, 0.111)
#         obj_rot_low = (-np.pi/3)
#         obj_rot_high = (np.pi/3)

#         self.hand_low = hand_low
#         self.hand_high = hand_high
#         self.obj_low = obj_low
#         self.obj_high = obj_high
#         self.goal_low = goal_low
#         self.goal_high = goal_high
#         self.retarget_step = 0   # like sawyer_assembly_peg_v2_all.py
#         self.not_grasp_final = True
#         self.obj_x = np.linspace(obj_low[0]+0.02, obj_high[0]-0.02, 5)
#         self.obj_y = np.linspace(obj_low[1]+0.02, obj_high[1]-0.02, 4)


#         super().__init__(
#             self.model_name,
#             hand_low=hand_low,
#             hand_high=hand_high,
#         )

#         self.init_config = {
#             'hammer_init_pos': np.array([0, 0.5, 0.0]),
#             'hand_init_pos': np.array([0, 0.4, 0.2]),
#         }
#         self.goal = self.init_config['hammer_init_pos']
#         self.hammer_init_pos = self.init_config['hammer_init_pos']
#         self.obj_init_pos = self.hammer_init_pos.copy()
#         self.hand_init_pos = self.init_config['hand_init_pos']
#         self.nail_init_pos = None

#         # self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
#         # allow randomize both the object position and goal position
#         self._random_reset_space = Box(
#             np.hstack((obj_low, goal_low, obj_rot_low)),
#             np.hstack((obj_high, goal_high, obj_rot_high)),
#         )
#         self.goal_space = Box(np.array(goal_low), np.array(goal_high))

#         # set test configurations for hand, camera, hammer, nail
#         self.obj_initial_cnt = 0
#         # Initialize hand
#         self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
#         self.hand_y = [0.52, 0.54, 0.56, 0.58]
#         self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]
#         # Initialize camera controller
#         self.camera_controller = CameraController(self)
#         self.theta_cam = [np.pi/21*i for i in range(1,21)]
#         # Initialize hammer
#         ...
#         self.theta_hammer = np.linspace(-np.pi/3, np.pi/3, 10)
#         # Initialize nail
#         ...


#     @property
#     def model_name(self):
#         return full_v2_path_for('sawyer_xyz/sawyer_hammer.xml')

#     @_assert_task_is_set
#     def evaluate_state(self, obs, action):
#         (
#             reward,
#             reward_grab,
#             reward_ready,
#             reward_success,
#             success
#         ) = self.compute_reward(action, obs)

#         info = {
#             'success': float(success),
#             'near_object': reward_ready,
#             'grasp_success': reward_grab >= 0.5,
#             'grasp_reward': reward_grab,
#             'in_place_reward': reward_success,
#             'obj_to_target': 0,
#             'unscaled_reward': reward,
#         }

#         return reward, info

#     def _get_id_main_object(self):
#         return self.unwrapped.model.geom_name2id('HammerHandle')

#     def _get_pos_objects(self):
#         return np.hstack((
#             self.get_body_com('hammer').copy(),
#             self.get_body_com('nail_link').copy()
#         ))

#     def _get_quat_objects(self):
#         return np.hstack((
#             self.sim.data.get_body_xquat('hammer'),
#             self.sim.data.get_body_xquat('nail_link')
#         ))

#     def _set_hammer_xyz(self, pos):
#         qpos = self.data.qpos.flat.copy()
#         qvel = self.data.qvel.flat.copy()
#         qpos[9:12] = pos.copy()
#         qvel[9:15] = 0
#         self.set_state(qpos, qvel)

#     def reset_model(self, set_xyz=None):
#         print('random_init=', self.random_init)
#         self._reset_hand()

#         # Set position of box & nail (these are not randomized)
#         self.sim.model.body_pos[self.model.body_name2id(
#             'box'
#         )] = np.array([0.24, 0.85, 0.0]) # configured in sawyer_hammer.xml
#         # Update _target_pos
#         self._target_pos = self._get_site_pos('goal')
#         # reset hammer & camera (default)
#         self.hammer_init_pos = np.array([
#             self.obj_x[self.obj_initial_cnt % 5], 
#             self.obj_y[self.obj_initial_cnt % 4], 
#             0])
#         self.hammer_init_rot = self.theta_hammer[self.obj_initial_cnt % 10]
#         self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)

#         self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])

#         # Randomize init position for hammer & camera
#         if self.random_init:
#             tmp = self._get_state_rand_vec() #(hammer_pos, target, hammer_rot)
#             while np.linalg.norm(tmp[:2] - tmp[3:5]) < 0.15 :  # too close to target
#                 tmp = self._get_state_rand_vec()
#             # self._target_pos = tmp[3:]
#             self.hammer_init_pos = tmp[:3]
#             self.hammer_init_rot = tmp[-1]
#             self.camera_controller.randomize_camera(method='cylindrical')
#             self.hand_init_pos = np.random.uniform(low=np.array([-0.4, 0.4, 0.1]), high=np.array([0.4, 0.6, 0.4]))
            
#         self.hand_init_pos[2] = max(self.hand_init_pos[2], 0.1)
#         self.nail_init_pos = self._get_site_pos('nailHead')
#         self.obj_init_pos = self.hammer_init_pos.copy()
#         self._set_hammer_xyz(self.hammer_init_pos)
#         self._set_obj_xyz(self.obj_init_pos)
#         self.sim.model.body_quat[self.model.body_name2id('hammer')] = random_z_rotation_quaternion(self.hammer_init_rot/np.pi*180)
#         self.init_tcp = self.tcp_center
#         self.init_left_pad = self.get_body_com('leftpad')
#         self.init_right_pad = self.get_body_com('rightpad')
#         self.sim.forward()
#         self.obj_initial_cnt += 1
#         print(f"hammer_init\npos=\n{self.hammer_init_pos}\nquat=\n{self.hammer_init_rot}\n")
#         return self._get_obs()

#     @staticmethod
#     def _reward_quat(obs):
#         # Ideal laid-down wrench has quat [1, 0, 0, 0]
#         # Rather than deal with an angle between quaternions, just approximate:
#         ideal = np.array([1., 0., 0., 0.])
#         error = np.linalg.norm(obs[7:11] - ideal)
#         return max(1.0 - error / 0.4, 0.0)

#     @staticmethod
#     def _reward_pos(hammer_head, target_pos):
#         pos_error = target_pos - hammer_head

#         a = 0.1  # Relative importance of just *trying* to lift the hammer
#         b = 0.9  # Relative importance of hitting the nail
#         lifted = hammer_head[2] > 0.02
#         in_place = a * float(lifted) + b * reward_utils.tolerance(
#             np.linalg.norm(pos_error),
#             bounds=(0, 0.02),
#             margin=0.2,
#             sigmoid='long_tail',
#         )

#         return in_place

#     def compute_reward(self, actions, obs):
#         hand = obs[:3]
#         hammer = obs[4:7]
#         hammer_quat = obs[7:11]
#         hammer_head = hammer + np.array([.16, .06, .0])
#         # consider rotate
#         hammer_head = rotate_local(hammer_quat, hammer, np.array([.16, .06, .0]))
#         # `self._gripper_caging_reward` assumes that the target object can be
#         # approximated as a sphere. This is not true for the hammer handle, so
#         # to avoid re-writing the `self._gripper_caging_reward` we pass in a
#         # modified hammer position.
#         # This modified position's X value will perfect match the hand's X value
#         # as long as it's within a certain threshold
#         hammer_threshed = hammer.copy()
#         threshold = SawyerHammerEnvV2.HAMMER_HANDLE_LENGTH / 2.0
#         if abs(hammer[0] - hand[0]) < threshold:
#             hammer_threshed[0] = hand[0]

#         reward_quat = SawyerHammerEnvV2._reward_quat(obs)
#         reward_grab = self._gripper_caging_reward(
#             actions, hammer_threshed,
#             object_reach_radius=0.01,
#             obj_radius=0.015,
#             pad_success_thresh=0.02,
#             xz_thresh=0.01,
#             high_density=True,
#         )
#         reward_in_place = SawyerHammerEnvV2._reward_pos(
#             hammer_head,
#             self._target_pos
#         )

#         reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
#         # Override reward on success. We check that reward is above a threshold
#         # because this env's success metric could be hacked easily
#         success = self.data.get_joint_qpos('NailSlideJoint') > 0.09
#         if success and reward > 5.:
#             reward = 10.0

#         return (
#             reward,
#             reward_grab,
#             reward_quat,
#             reward_in_place,
#             success,
#         )

#     # =====copy from sawyer_assembly_all_v2.py=====
#     def set_new_target(self, v, d):
#         """
#         possible functions:
#             if v is set , move direction d
#             if v is not set , randomly move  
#         """
#         if v is not None:
#             self._target_pos = np.clip(self._target_pos + v*d[0], self.goal_low, self.goal_high)
#             self._target_pos[-1] = self.sim.model.body_pos[self.model.body_name2id("tablelink")][-1]+0.1

#             if self._target_pos[0] == self.goal_low[0] or self._target_pos[0] == self.goal_high[0]:
#                 d[0][0] = -d[0][0]
#             if self._target_pos[1] == self.goal_low[1] or self._target_pos[1] == self.goal_high[1]:
#                 d[0][1] = -d[0][1]

#             peg_pos = self._target_pos - np.array([0., 0., 0.05])
#             self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
#             self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos

#             return d


#     def set_new_obj(self, v, d):
#         if v != 0:
#             self.obj_init_pos = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)
#             self.obj_init_pos[-1] = self.sim.model.body_pos[self.model.body_name2id("tablelink")][-1]+0.02
#             self._set_obj_xyz_move(self.obj_init_pos)

#             if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
#                 d[0][0] = -d[0][0]
#             if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
#                 d[0][1] = -d[0][1]

#             return d

#     def rotate_obj(self, state=None, v=0, d=None):
#         """

#         """
#         if v != 0:
#             self._set_obj_xyz(self.obj_init_pos)
#             drot = math.degrees(2*math.asin(self.sim.model.body_quat[self.model.body_name2id('hammer')][-1]))
#             self.sim.model.body_quat[self.model.body_name2id('hammer')] = random_z_rotation_quaternion(drot+v*d)
            
#             self.sim.forward()
#             # SawyerNutAssemblyAllEnvV2.delta_xyz = self.data.site_xpos[self.model.site_name2id('RoundNut-8')] - self.data.site_xpos[self.model.site_name2id('RoundNut')]
#             theta = 2*math.asin(self.sim.model.body_quat[self.model.body_name2id('hammer')][-1])/math.pi*180

#             if theta < -60 or theta > 60:
#                 d = -d

#             return d

#         else:
#             raise ValueError("v should not be 0, please set a valid v value")
#             exit(0)   


#     def set_rotate_and_new_obj(self, state=None, v_rotate=0, d_rotate=None, v_move=0, d_move=None):
#         # print(v_rotate, v_move)
#         if v_rotate != 0 and v_move != 0:
#             d_move = self.set_new_obj(v=v_move, d=d_move)
#             d_rotate = self.rotate_obj(v=v_rotate, d=d_rotate)
#         elif v_rotate != 0:
#             d_rotate = self.rotate_obj(v=v_rotate, d=d_rotate)
#         elif v_move != 0:
#             d_move = self.set_new_obj(v=v_move, d=d_move)
#         else:
#             raise ValueError("v_rotate and v_move should not be both 0, please set a valid v value")
#             exit(0)  

#         return d_rotate, d_move