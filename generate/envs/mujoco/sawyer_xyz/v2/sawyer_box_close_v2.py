import numpy as np
import math
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_hand_camera_v2 import CameraController # use camera controller from pick place env
from metaworld.policies.sawyer_assembly_v2_all_policy import rotate_local
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_assembly_peg_v2_all import random_z_rotation_quaternion


class SawyerBoxCloseEnvV2(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        self.obj_low = obj_low = (-0.3, 0.4, 0.02)
        self.obj_high = obj_high = (0.3, 0.5, 0.02)
        self.goal_low = goal_low = (-0.25, 0.75, 0.133)
        self.goal_high = goal_high = (0.25, 0.85, 0.133)
        self.obj_theta_range = (-np.pi/4+np.pi/2, np.pi/4+np.pi/2)
        self.theta_range = (0, np.pi)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.55, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_to_obj_init = None

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self._random_reset_space = Box(
            np.hstack((obj_low, self.obj_theta_range[0], goal_low)),
            np.hstack((obj_high, self.obj_theta_range[1],goal_high)),
        )
        self.camera_controller = CameraController(self)
        
        self.theta_cam = [self.theta_range[0]*i/21 + self.theta_range[1]*(21-i)/21  for i in range(1,21)]

        self.obj_x = np.linspace(obj_low[0]+0.02, obj_high[0]-0.02, 5)
        self.obj_y = np.linspace(obj_low[1]+0.02, obj_high[1]-0.02, 4)

        self.hand_x = np.linspace(0.32, 0.48, 10)#[-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]
        
        box_x = np.linspace(goal_low[0]+0.02, goal_high[0]-0.02,5)
        box_y = np.linspace(goal_low[1]+0.02, goal_high[1]-0.02,4)
        self.test_config = {
            'obj_init_pos': np.array([[x,y,0] for x in self.obj_x for y in self.obj_y]),
            'obj_init_rot': np.linspace(*self.obj_theta_range, 22)[1:-1],
            'goal_init_pos': np.array([[x,y,0] for x in box_x  for y in box_y]),
            'hand_init_pos': np.array([[self.hand_x[i%10], self.hand_y[i%4], self.hand_z[i%5]] for i in range(20)]),
        }
        self.obj_initial_cnt = 0
        self.retarget_step = 90 # modify it

        self.ROTATE_DISABLE = False
        self.CAMERA_DISABLE = False
        self.MOVE_OBJ_DISABLE = False
        self.MOVE_TARGET_DISABLE = False

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_box.xml')

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

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('BoxHandleGeom')

    def _get_pos_objects(self):
        return self.get_body_com('top_link')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('top_link')
    
    def _set_obj_quat(self, quat):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self, set_xyz = True):
        """
            obj: boxbodytop
            target: boxbody
        """
        print(self.random_init)
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        box_height = self.get_body_com('boxbody')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.25:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.obj_init_rot = goal_pos[3]
            self._target_pos = goal_pos[-3:]
            self.hand_init_pos = np.random.uniform(low=np.array([0.3, 0.5, 0.2]), high=np.array([0.5, 0.6, 0.4]))
            self.camera_controller.randomize_camera(method='cylindrical', theta_range = self.theta_range)
        else:
            self.obj_init_pos = self.test_config['obj_init_pos'][self.obj_initial_cnt % 20]
            self.obj_init_rot = self.test_config['obj_init_rot'][self.obj_initial_cnt % 20]
            self._target_pos = self.test_config['goal_init_pos'][self.obj_initial_cnt % 20]
            self.hand_init_pos = self.test_config['hand_init_pos'][self.obj_initial_cnt % 20]
            self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)

            self.obj_initial_cnt += 1
        
        print(f"set boxbody{self.sim.model.body_pos[self.model.body_name2id('boxbody')]} to {np.concatenate((self._target_pos[:2], [box_height]))}")
        self.sim.model.body_pos[self.model.body_name2id('boxbody')] = np.concatenate((self._target_pos[:2], [box_height]))
        self._set_obj_xyz(self.obj_init_pos)
        
        self.obj_init_quat = random_z_rotation_quaternion(np.degrees(self.obj_init_rot))
        
        if self.ROTATE_DISABLE:
            self.obj_init_rot = 0
            self.obj_init_quat = np.array([1, 0, 0, 0])
            print("ROTATE_DISABLE")
        else:
            self.upd_rotate()
        
        self.delta_hight = -self.data.get_body_xpos('boxbodytop')[-1]
        self._reset_hand()
        self.delta_hight += self.data.get_body_xpos('boxbodytop')[-1]
        print(f"get boxbody{self.sim.model.body_pos[self.model.body_name2id('boxbody')]}")
        print(f"get toplink pos:{self.get_body_com('top_link')}")

        return self._get_obs()
    
    def upd_rotate(self):
        """
            update _lever_pos, _target_pos according to self.obj_init_quat
        """
        # print(np.degrees(self.obj_init_rot), self.obj_init_quat)
        # self.sim.model.body_quat[
        #     self.model.body_name2id('boxbodytop')] = self.obj_init_quat
        # self.sim.forward()
        self._set_obj_quat(self.obj_init_quat)


    def set_new_obj(self, state=None, v=None, d=None):
        if self.MOVE_OBJ_DISABLE:
            return
        
        if not (state is None):
            self._target_pos = state
            return
        elif v != 0 and not(v is None):
            # tmp = self._get_pos_objects()
            tmp=self.data.qpos.flat.copy()[9:12]
            self.obj_init_pos = np.clip(tmp+d[0]*v, self.obj_low, self.obj_high)
            self.obj_init_pos[2] = tmp[2]
            self._set_obj_xyz(self.obj_init_pos)
            # self.data.get_body_xpos('boxbodytop')
            # print(self.obj_init_pos, "vs",self._get_pos_objects())
            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]
            if self.obj_init_pos[2] == self.obj_low[2] or self.obj_init_pos[2] == self.obj_high[2]:
                d[0][2] = -d[0][2]
            return d
        
    def rotate_obj(self, state=None, v=0, d=None):
        if self.ROTATE_DISABLE:
            return
        
        if v != 0:
            self.obj_init_rot = np.clip(self.obj_init_rot + np.deg2rad(v*d*0.1), *self.obj_theta_range) # fps=10
            self.obj_init_quat = random_z_rotation_quaternion(np.degrees(self.obj_init_rot))
            self.upd_rotate()
            

            if self.obj_init_rot in self.obj_theta_range:
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

    def set_new_target(self, v, d):
        if self.MOVE_TARGET_DISABLE:
            return
        """
        possible functions:
            if v is set , move direction d
            if v is not set , randomly move  
        """
        if v is not None:
            self._target_pos = np.clip(self._target_pos + v*d[0], self.goal_low, self.goal_high)
            # self._target_pos[-1] = self.sim.model.body_pos[self.model.body_name2id("tablelink")][-1]+0.1

            if self._target_pos[0] == self.goal_low[0] or self._target_pos[0] == self.goal_high[0]:
                d[0][0] = -d[0][0]
            if self._target_pos[1] == self.goal_low[1] or self._target_pos[1] == self.goal_high[1]:
                d[0][1] = -d[0][1]

            # peg_pos = self._target_pos - np.array([0., 0., 0.05])
            self.sim.model.body_pos[self.model.body_name2id('boxbody')][:2] = self._target_pos[:2]
            self.sim.forward()
            # self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
            # self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos

            return d

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_quat(obs):
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = min(np.linalg.norm(obs[7:11] - ideal), np.linalg.norm(obs[7:11] - np.array([0.707, 0, 0, -0.707])))
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(obs, target_pos):
        hand = obs[:3]
        lid = obs[4:7] + np.array([.0, .0, .02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - lid),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1., 1., 3.])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * error_scale),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid='long_tail',
        )
        # print("ready_to_lift=", ready_to_lift)
        return ready_to_lift, lifted

    def compute_reward(self, actions, obs):
        reward_grab = SawyerBoxCloseEnvV2._reward_grab_effort(actions)
        reward_quat = SawyerBoxCloseEnvV2._reward_quat(obs)
        reward_steps = SawyerBoxCloseEnvV2._reward_pos(obs, self._target_pos)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success

        
        w, x, y, z = obs[7:11]
        yaw = abs(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))

        ideal_height = 0.06 + 0.016

        success = abs(ideal_height - obs[6]) < 0.04 and np.linalg.norm(obs[4:6] - self._target_pos[:2]) < 0.02 and min(abs(np.degrees(yaw)+90)+1000000,abs(np.degrees(yaw)-90)) < 5
        if success:
            reward = 10.0

        # STRONG emphasis on proper lid orientation to prevent reward hacking
        # (otherwise agent learns to kick-flip the lid onto the box)
        reward *= reward_quat

        return (
            reward,
            reward_grab,
            *reward_steps,
            success,
        )
