import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation
import math

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

task_type = 'theta'

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



class SawyerPushHandCameraEnvV2(SawyerXYZEnv):
    TARGET_RADIUS=0.05

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        goal_low = (-0.1, 0.8, 0.01)
        goal_high = (0.1, 0.9, 0.02)               
        # obj_low = (-0.1, 0.6, 0.02)
        # obj_high = (0.1, 0.7, 0.02)
        # obj_low = (-0.3, 0.4, 0.02)
        # obj_high = (0.2, 0.9, 0.02)
        # tmp = [[-0.2, 0.5, 0.02], [-0.2, 0.8, 0.02], [0.1, 0.5, 0.02], [0.1, 0.8, 0.02]]
        obj_low = (-0.2, 0.5, 0.02)
        obj_high = (0.1, 0.8, 0.02)
        # obj_low = (-0.1, 0.55, 0.02)
        # obj_high = (0.1, 0.75, 0.02)
        # obj_low = (-0.2, 0.6, 0.02)
        # obj_high = (0.1, 0.7, 0.02)
        # obj_low = (-0.2, 0.55, 0.02)
        # obj_high = (0.1, 0.75, 0.02)

        self.goal_low = goal_low
        self.goal_high = goal_high
        self.obj_low = obj_low
        self.obj_high = obj_high

        if  obj_low[0] == -0.2 and obj_low[1] == 0.55:#3*2
            self.obj_x = [-0.2, -0.125, -0.05, 0.025, 0.1]
            self.obj_y = [0.55, 0.64, 0.7, 0.75]
        elif obj_low[0] == -0.1 and obj_low[1] == 0.55:#2*2
            self.obj_x = [-0.1, -0.05, 0.0, 0.05, 0.1]
            self.obj_y = [0.55, 0.64, 0.7, 0.75]
        elif obj_low[0] == -0.2 and obj_low[1] == -0.6:#3*1
            self.obj_x = [-0.2, -0.125, -0.05, 0.025, 0.1]
            self.obj_y = [0.6, 0.63, 0.66, 0.7]
        elif obj_low[0] == -0.1 and obj_low[1] == 0.6:#3*1
            self.obj_x = [-0.1, -0.05, 0.0, 0.05, 0.1]
            self.obj_y = [0.6, 0.63, 0.66, 0.7]
        else:
            # self.obj_x = [-0.2, -0.125, -0.025, 0.05, 0.1]
            # self.obj_y = [0.5, 0.6, 0.7, 0.8]
            self.obj_x = [-0.18, -0.115, -0.05, 0.015, 0.08]
            self.obj_y = [0.52, 0.607, 0.693, 0.78]

        self.goal_points =[
                
                (-0.1, 0.8, 0.01), (-0.1, 0.8, 0.02), (-0.1, 0.9, 0.01), (-0.1, 0.9, 0.02),
                (0.1, 0.8, 0.01), (0.1, 0.8, 0.02), (0.1, 0.9, 0.01), (0.1, 0.9, 0.02),
                
                
                (0.0, 0.85, 0.013), (-0.05, 0.85, 0.015), (0.05, 0.85, 0.017),
                
                
                (-0.075, 0.8, 0.015), (0.075, 0.9, 0.015), 
                (-0.025, 0.85, 0.015), (0.025, 0.85, 0.015),
                
                
                (-0.05, 0.8, 0.012), (0.05, 0.9, 0.014),
                (0.0, 0.83, 0.016), (0.0, 0.87, 0.018),
                
                
                (-0.08, 0.82, 0.016), (0.08, 0.88, 0.012)
        ]

        self.hand_x = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.25, -0.05, 0.25] 
        self.hand_y = [0.5, 0.55, 0.6, 0.65]  
        self.hand_z = [0.2, 0.25, 0.3, 0.35, 0.4] 
        self.theta_cam = [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 
                         5*np.pi/6, np.pi, 7*np.pi/6, 5*np.pi/4, 4*np.pi/3, 3*np.pi/2,
                         5*np.pi/3, 7*np.pi/4, 11*np.pi/6, 0, np.pi/8, 3*np.pi/8, 
                         5*np.pi/8, 7*np.pi/8]  
        
        self.obj_initial_cnt = 0

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.move_object = True
        self.retarget = False


        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0., 0.6, 0.02]),
            'hand_init_pos': np.array([0., 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.02])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self._target_pos = self.goal.copy()

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))


        self.camera_controller = CameraController(self)

        self.num_resets = 0
        


    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_push_v2.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(
                self.touching_main_object and
                (tcp_opened > 0) and
                (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self, set_xyz=False):
        self._reset_hand()
        if set_xyz:
            self._target_pos = self.goal.copy()
            # self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])     #？？？
            self.obj_init_angle = self.init_config['obj_init_angle']

            if self.random_init:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
                while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                    goal_pos = self._get_state_rand_vec()
                    # goal_pos[1] = 0.5 + (0.8 - 0.5) * beta.rvs(1, 3, size=1)[0]
                    self._target_pos = goal_pos[3:]
                self._target_pos = goal_pos[-3:]
                self.obj_init_pos = goal_pos[:3]
                self.hand_init_pos = np.random.uniform(low=np.array([-0.3, 0.5, 0.2]), high=np.array([0.3, 0.6, 0.4]))
                self.camera_controller.randomize_camera(method='cylindrical')
                
            else:
                self.obj_init_pos = [self.obj_x[(self.obj_initial_cnt-1)%5], self.obj_y[(self.obj_initial_cnt-1)%4], 0.02]
                self._target_pos = np.array(list(self.goal_points[(self.obj_initial_cnt-1)%20]))
                self.obj_initial_cnt += 1
                self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])
                self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)

        
        else:
            self._target_pos = self.goal.copy()
            self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
            self.obj_init_angle = self.init_config['obj_init_angle']

        # 记录初始夹爪位置
        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1
        return self._get_obs()

    # def reset_model(self):
    #     self._reset_hand()
    #     self._target_pos = self.goal.copy()
    #     self.obj_init_pos = np.array(self.fix_extreme_obj_pos(self.init_config['obj_init_pos']))
    #     self.obj_init_angle = self.init_config['obj_init_angle']

    #     if self.random_init:
    #         goal_pos = self._get_state_rand_vec()
    #         self._target_pos = goal_pos[3:]
    #         while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
    #             goal_pos = self._get_state_rand_vec()
    #             self._target_pos = goal_pos[3:]
    #         self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
    #         self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

    #     self._set_obj_xyz(self.obj_init_pos)
    #     self.num_resets += 1

    #     return self._get_obs()



    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp = self.tcp_center
        tcp_opened = obs[3]
        target = self._target_pos
        tcp_to_obj = np.linalg.norm(obj - tcp)
        target_to_obj = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True
        )
        
        reward = 2 * object_grasped

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + reward + 5. * in_place
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


    def set_new_obj(self, obj_pos):

        self.obj_init_pos = np.array(obj_pos)
        self._set_obj_xyz(self.obj_init_pos)

    def set_new_target(self, target_pos):

        self._target_pos = np.array(target_pos)

    def set_camera_pos(self, camera_pos=None, method='preset'):

        if camera_pos is not None:
            self.camera_controller.set_camera_pose(camera_pos)
        else:
            self.camera_controller.randomize_camera(method=method)

    def set_new_hand_pos(self, hand_pos):

        self.hand_init_pos = np.array(hand_pos)
        self._reset_hand()

