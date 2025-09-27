import numpy as np
from gym.spaces import Box
import math

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

class CameraController:
    def __init__(self, env, camera_name='corner2', task_name=''):
        self.env = env
        self.sim = env.sim
        self.camera_name = camera_name
        self.target_point = np.array([0.0, 0.6, 0.2]) 
        self.r, self.theta, self.z = 0, 0, 0
        
        try:
            self.camera_id = self.sim.model.camera_name2id(camera_name)
        except:
            self.camera_id = 0
    
    def look_at_euler(self, camera_pos, target_pos):
        dx = target_pos[0] - camera_pos[0]
        dy = target_pos[1] - camera_pos[1]
        dz = target_pos[2] - camera_pos[2]
        
        horizontal_dist = np.sqrt(dx*dx + dy*dy)
        
        yaw = np.arctan2(dy, dx)
        
        pitch = np.arctan2(dz, horizontal_dist) 

        roll = 0

        # print(yaw, pitch, roll)
        # yaw, pitch, roll = 3.9, 2.3, 0.6
        
        return np.array([roll, pitch, yaw])

    def look_at_matrix(self, camera_pos, target_pos, up=np.array([0, 0, -1])):
        eye = np.array(camera_pos, dtype=float)
        target = np.array(target_pos, dtype=float)
        up = np.array(up, dtype=float)
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:

            if abs(forward[2]) > 0.9:
                up = np.array([1, 0, 0])
            else:
                up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
        
        right = right / right_norm
        

        up_corrected = np.cross(right, forward)
        

        R_matrix = np.array([
            right,           
            up_corrected,    
            -forward        
        ]).T
        
        from scipy.spatial.transform import Rotation as R_scipy
        r = R_scipy.from_matrix(R_matrix)
        quat_xyzw = r.as_quat()
        
        # 转换为MuJoCo的wxyz格式
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return quat_wxyz
    
    def euler_to_quaternion(self, euler):
        roll, pitch, yaw = euler
        
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5) 
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        # print(np.array([w, x, y, z]))
        return np.array([w, x, y, z])
        # return np.array([-0.395, 0.252, -0.435, 0.765])/0.997
    
    def matrix_to_quaternion(self, rotation_matrix):
        r = R.from_matrix(rotation_matrix)
        quat_xyzw = r.as_quat()  
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return quat_wxyz
    
    def set_camera_pose(self, position, quaternion=None):
        position = np.array(position, dtype=float)
        
        dist_to_target = np.linalg.norm(position - self.target_point)
        if dist_to_target < 0.1:
            return False
    
        if position[2] < 0:
            position[2] = 0.1 
        
        if quaternion is None:
            euler = self.look_at_euler(position, self.target_point)
            # quaternion = self.euler_to_quaternion(euler)
            quaternion = self.look_at_matrix(position, self.target_point)
        
        quat_norm = np.linalg.norm(quaternion)
        if abs(quat_norm - 1.0) > 1e-3:
            quaternion = quaternion / quat_norm
        
        self.env.model.cam_pos[self.camera_id] = position
        self.env.model.cam_quat[self.camera_id] = quaternion

        return True

    def sample_spherical_position(self, center=None, radius_range=(0.8, 1.5), 
                                 theta_range=(0, 2*np.pi), phi_range=(np.pi/6, 2*np.pi/3)):
        if center is None:
            center = self.target_point
        
        r = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(theta_range[0], theta_range[1])
        phi = np.random.uniform(phi_range[0], phi_range[1])
        
        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)
        
        z = max(z, 0.1)
        
        return np.array([x, y, z])
    
    def sample_cylindrical_position(self, center=None, radius_range=(0.8, 1.5), 
                                   height_range=(0.4, 1.0), theta_range=(-np.pi/3, np.pi*2/3)):
        if center is None:
            center = self.target_point
        
        r = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(theta_range[0], theta_range[1])
        z = np.random.uniform(height_range[0], height_range[1])


        if task_type == 'height':
            r = math.sqrt(0.6**2+0.295**2)
            # z = 0.6
            theta = np.pi / 3
        elif task_type == 'r':
            # r = math.sqrt(0.6**2+0.295**2)
            z = 0.6
            theta = np.pi / 3
        elif task_type == 'theta':
            r = math.sqrt(0.6**2+0.295**2)
            z = 0.6
            # theta = np.pi / 3

        self.r = r
        self.theta = theta
        self.z = z
        
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + z
        
        z = max(z, 0.1)
        
        return np.array([x, y, z])
    
    def sample_preset_positions(self):
        preset_positions = [
            np.array([0.5, 1.2, 0.8]), 
            np.array([-0.5, 1.2, 0.8]), 
            np.array([0, 1.2, 0.8]),  
            np.array([1.0, 0.6, 0.6]),  
            np.array([-1.0, 0.6, 0.6]), 
            np.array([0.0, 0.6, 1.5]), 
        ]
        
        return preset_positions[np.random.randint(len(preset_positions))]
    
    def randomize_camera(self, method='spherical', **kwargs):
        if method == 'spherical':
            position = self.sample_spherical_position(**kwargs)
        elif method == 'cylindrical':
            position = self.sample_cylindrical_position(**kwargs)
        elif method == 'preset':
            position = self.sample_preset_positions()
        else:
            raise ValueError(f"{method}")
        
        self.set_camera_pose(position)
        return position

    def set_cylindrical_camera(self, center=None, radius=None, height=None, theta=None):

        if center is None:
            center = self.target_point

        r = radius
        theta = theta
        z = height

        self.r = r
        self.theta = theta
        self.z = z
        
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + z
        
        z = max(z, 0.1)
        
        position = np.array([x, y, z])
        
        self.set_camera_pose(position)
        return position
    
    def get_camera_info(self):
        pos = self.sim.model.cam_pos[self.camera_id].copy()
        quat = self.sim.model.cam_quat[self.camera_id].copy()
        
        return {
            'position': pos,
            'quaternion': quat,
            'target': self.target_point,
            'distance_to_target': np.linalg.norm(pos - self.target_point)
        }



class SawyerBasketballEnvV2(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN = 0.06
    TARGET_RADIUS = 0.08

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        # obj_low = (-0.1, 0.6, 0.0299)
        # obj_high = (0.1, 0.7, 0.0301)
        # goal_low = (-0.1, 0.85, 0.)
        # goal_high = (0.1, 0.9+1e-7, 0.)

        obj_low = (-0.3, 0.5, 0.0299)
        obj_high = (0.3, 0.7, 0.0301)
        goal_low = (-0.3, 0.8, 0.)
        goal_high = (0.3, 0.9+1e-7, 0.)

        self.obj_low = obj_low
        self.obj_high = obj_high
        self.goal_low = goal_low
        self.goal_high = goal_high

        self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]

        self.obj_x = [-0.28, -0.15, 0, 0.15, 0.28]
        self.obj_y = [0.52, 0.57, 0.62, 0.68]
        self.goal_x = [-0.28, -0.15, 0, 0.15, 0.28]
        self.goal_y = [0.82, 0.84, 0.86, 0.88]

        self.theta_cam = [np.pi/21*i/2 for i in range(1,21)]

        self.obj_initial_cnt = 0

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.03], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, 0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_pos = self.goal.copy()

        self.retarget = False
        self.move_object = True

        self.move_step = 30
        self.early_not_move = False
        self.retarget_max_step = 140

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501])
        )

        self.camera_controller = CameraController(self)

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_basketball.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(
                (tcp_open > 0) and
                (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info


    def set_new_target(self, state=None, v=None, d=None):
        if not (state is None):
            self._target_pos = state
            return
        elif v != 0 and not(v is None):
            # self._target_pos = np.clip(self._target_pos+d[0]*v, self.goal_low, self.goal_high)
            self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = np.clip(self.sim.model.body_pos[self.model.body_name2id('basket_goal')]+d[0]*v, self.goal_low, self.goal_high)
            self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]

            tmp = self.sim.model.body_pos[self.model.body_name2id('basket_goal')]
            if tmp[0] == self.goal_low[0] or tmp[0] == self.goal_high[0]:
                d[0][0] = -d[0][0]
            if tmp[1] == self.goal_low[1] or tmp[1] == self.goal_high[1]:
                d[0][1] = -d[0][1]
            if tmp[2] == self.goal_low[2] or tmp[2] == self.goal_high[2]:
                d[0][2] = -d[0][2]
            return d
        while True:
            # print("reset target", self._target_pos)
            tmp = np.array([
                np.random.uniform(self.goal_low[0], self.goal_high[0]),  # 第一个元素：[-0.3, 0.3]
                np.random.uniform(self.goal_low[1], self.goal_high[1]),   # 第二个元素：[0.6, 0.8]
                np.random.uniform(self.goal_low[2], self.goal_high[2])      # 第三个元素：[0, 0.2]
            ])
            # print(tmp, self.sim.model.body_pos[self.model.body_name2id('basket_goal')])
            # print("*********")
            # self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = np.array([0, 0.8, 0])
            # self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]
            # # self.sim.forward()
            # break
            if np.linalg.norm(tmp - self.sim.model.body_pos[self.model.body_name2id('basket_goal')]) >= 0.3:#0.33541019662496846/2:# and np.linalg.norm(tmp - self._target_pos) >= 0.1:
                self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = tmp
                # self.sim.forward()
                self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]
                self._set_obj_xyz_basketball(self.get_body_com('basketball'))      
                # self.sim.forward()
                break

        # self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = np.array([0, 0.8, 0])
        # self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]


    def set_new_obj(self, state=None, v=0, d=None):
        if v != 0:
            self.obj_init_pos = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)
            # print("target, ", self._target_pos)
            self._set_obj_xyz_basketball_move(self.obj_init_pos, target=self._target_pos)
            # print(d)
            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]
            return d
        elif state:
            self.obj_init_pos = np.clip([self.get_body_com('obj')[0]+np.random.randn()*0.06, 0.8 - np.random.rand()*0.05 if self.obj_init_pos[1] >= 0.8 else self.get_body_com('obj')[1]+np.random.rand()*0.005, 0.02], self.obj_low, self.obj_high)
            self._set_obj_xyz_basketball(self.obj_init_pos)            
        else:
            while True:
                # print("obj", self.obj_init_pos)
                tmp = np.array([
                    np.random.uniform(self.obj_low[0], self.obj_high[0]),  # 第一个元素：[-0.3, 0.3]
                    np.random.uniform(self.obj_low[1], self.obj_high[1]),   # 第二个元素：[0.6, 0.8]
                    np.random.uniform(self.obj_low[2], self.obj_high[2])      # 第三个元素：[0, 0.2]
                ])
                if np.linalg.norm(tmp - self.obj_init_pos) >= 0.223606797749979/2:
                    self.obj_init_pos = tmp
                    # self._set_obj_xyz(self.obj_init_pos)
                    self._set_obj_xyz_basketball_move(self.obj_init_pos, target=self._target_pos)
                    break    


    def set_camera_pos(self, v_theta_cam=0):
        theta = self.camera_controller.theta
        r = self.camera_controller.r
        h = self.camera_controller.z
        theta = np.clip(theta+v_theta_cam, 0, np.pi)
        self.camera_controller.set_cylindrical_camera(radius=r, theta=theta, height=h)
        if theta == 0 or theta == np.pi:
            v_theta_cam = -v_theta_cam
        return v_theta_cam


    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('bsktball')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('bsktball')

    def reset_model(self, set_xyz=False):
        self._reset_hand()
        # print(set_xyz)

        if set_xyz:
            if self.random_init:
                goal_pos = self._get_state_rand_vec()
                basket_pos = goal_pos[3:]
                while np.linalg.norm(goal_pos[:2] - basket_pos[:2]) < 0.15:
                    goal_pos = self._get_state_rand_vec()
                    basket_pos = goal_pos[3:]
                self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
                # print("basket: ", basket_pos)
                self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = np.concatenate((basket_pos[:2], [0]))
                # self._target_pos = np.concatenate((self.data.site_xpos[self.model.site_name2id('goal')][:2], [0]))
                self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]
                # self._target_pos = np.concatenate((basket_pos[:2], [0]))
                # print("basket: ", basket_pos, self.data.site_xpos[self.model.site_name2id('goal')])
                self.hand_init_pos = np.random.uniform(low=np.array([-0.3, 0.5, 0.2]), high=np.array([0.3, 0.6, 0.4]))
                self.camera_controller.randomize_camera(method='cylindrical')
            else:
                self.obj_init_pos = np.array([self.obj_x[(self.obj_initial_cnt-1)%5], self.obj_y[(self.obj_initial_cnt-1)%4], 0.02])
                self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = [self.goal_x[(self.obj_initial_cnt-1)%5], self.goal_y[(self.obj_initial_cnt-1)%4], 0.02]
                self._target_pos = np.array([self.goal_x[(self.obj_initial_cnt-1)%5], self.goal_y[(self.obj_initial_cnt-1)%4], 0.02])
                self.obj_initial_cnt += 1
                self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])
                self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)

        else:
            self._target_pos = self._target_pos
            # self.obj_init_pos = self.fix_extreme_obj_pos(self.obj_init_pos)
            self.obj_init_angle = self.init_config['obj_init_angle']
            # basket_pos = self.goal.copy()
            # self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = self._target_pos
            self._target_pos = self._target_pos

        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com('leftpad')
        self.init_right_pad = self.get_body_com('rightpad')
        self._set_obj_xyz_basketball(self.obj_init_pos)
        # print("target", self._target_pos)
        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        # Force target to be slightly above basketball hoop
        target = self._target_pos.copy()
        target[2] = 0.3

        # Emphasize Z error
        scale = np.array([1., 1., 2.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.025,
            pad_success_thresh=0.06,
            xz_thresh=0.005,
            high_density=True
        )
        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > self.obj_init_pos[2]:
            object_grasped = 1
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > self.obj_init_pos[2]:
            reward += 1. + 5. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
