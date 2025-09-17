import numpy as np
from gym.spaces import Box
import random
import math
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

def update_target_pos(window_quat, pos, local_offset):
    quat_scipy = np.roll(window_quat, -1)  # 转换为Scipy格式 (x,y,z,w)
    R = Rotation.from_quat(quat_scipy).as_matrix()
    global_offset = R.dot(local_offset)
    if not (pos is None):
        return pos + global_offset
    else:
        return global_offset

def random_z_rotation_quaternion(degrees=None):
    if degrees:
        degrees = degrees
    else:
        degrees = random.uniform(-180, 0)        # 生成0-180随机角度
    theta = math.radians(degrees)           # 转为弧度
    w = math.cos(theta / 2)                 # 实部计算
    z = math.sin(theta / 2)                 # z虚部分量
    return np.array([w, 0.0, 0.0, z])       # [w, x, y, z]格式

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
                                   height_range=(0.4, 1.0), theta_range=(-np.pi/2, np.pi/2)):
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
            np.array([0.5, 1.2, 0.8]),   # 右后方
            np.array([-0.5, 1.2, 0.8]),  # 左后方
            np.array([0, 1.2, 0.8]),   # 右后方
            np.array([1.0, 0.6, 0.6]),   # 右侧
            np.array([-1.0, 0.6, 0.6]),  # 左侧
            np.array([0.0, 0.6, 1.5]),   # 正上方
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
            raise ValueError(f"未知的采样方法: {method}")
        
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
        
        # 转换为笛卡尔坐标
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + z
        
        # 确保摄像机不会在地面以下
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



class SawyerFaucetOpenEnvV2(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.2, 0.6, 0.0)
        obj_high = (0.2, 0.85, 0.0)

        self.obj_low = (-0.2, 0.6, 0.0)
        self.obj_high = (0.2, 0.85, 0.0)
        self._handle_length = 0.175
        self._target_radius = 0.07

        self.obj_x = [-0.18, -0.08, 0, 0.08, 0.18]
        self.obj_y = [0.62, 0.66, 0.79, 0.83]
        self.obj_theta = [-178+176/20*i for i in range(0,21)]
        self.theta_cam = [np.pi/21*i-np.pi/2 for i in range(1,21)]

        self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]

        self.obj_initial_cnt = 0

        self.relocate_distance = 0.25

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.0]),
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.camera_controller = CameraController(self)


        self.move_step = 35
        self.retarget = False
        self.early_not_move = False
        self.move_object = True

        self._target_pos = np.array([0, 0.6, 0.2])

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_faucet.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, tcp_to_obj, _, target_to_obj, object_grasped,
         in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= 0.07),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [('goal_open', self._target_pos),
                ('goal_close', np.array([10., 10., 10.]))]

    def _get_pos_objects(self):
        return self._get_site_pos('handleStartOpen') + np.array(
            [0., 0., -0.01])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('faucetBase')

    def set_new_target(self, state=None, v=0, d=None):
        pass

    def set_new_obj(self, state=None, v=0, d=None):
        print("move object")
        if v != 0:
            self.obj_init_pos = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)
            self.sim.model.body_pos[self.model.body_name2id(
                'faucetBase')] = self.obj_init_pos
            self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125]))
            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]
            return d
        else:
            self.obj_init_pos = self._get_state_rand_vec()
            # Set mujoco body to computed position
            self.sim.model.body_pos[self.model.body_name2id(
                'faucetBase')] = self.obj_init_pos
            self.sim.model.body_quat[self.model.body_name2id(
                'faucetBase')] = random_z_rotation_quaternion()
            self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125])) #(window_quat, pos, local_offset)
    
 
    def rotate_obj(self, state=None, v=0, d=None):
        print("rotate object")

        if v != 0:
            self.sim.model.body_quat[self.model.body_name2id('faucetBase')] = random_z_rotation_quaternion()
            self.sim.forward()
            self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125])) #(window_quat, pos, local_offset)
            theta = 2*math.asin(self.sim.model.body_quat[self.model.body_name2id('faucetBase')][-1])/math.pi*180
            if theta > 0 or theta < -180:
                d = -d
            return d
        else:
            exit(0)  

    def set_camera_pos(self, v_theta_cam=0, v_r_cam=0.0, v_h_cam=0.0):
        # radius_range=(0.8, 1.5), height_range=(0.4, 1.0), theta_range=(0, np.pi))
        # r = np.clip(self.camera_controller.r+v_r_cam, 0.8, 1.5)
        # theta = np.clip(self.camera_controller.theta+v_theta_cam, 0, np.pi)
        # h = np.clip(self.camera_controller.z+v_h_cam, 0.4, 1.0)
        theta = self.camera_controller.theta
        r = self.camera_controller.r
        h = self.camera_controller.z
        theta = np.clip(theta+v_theta_cam, -np.pi/3, np.pi*2/3)

        self.camera_controller.set_cylindrical_camera(radius=r, theta=theta, height=h)
        if theta == -np.pi/3 or theta == np.pi*2/3:
            v_theta_cam = -v_theta_cam

        return v_theta_cam, 0, 0

    def set_rotate_and_new_obj(self, state=None, v_rotate=0, d_rotate=None, v_move=0, d_move=None):
        if v_move != 0:

            self.obj_init_pos = np.clip(self.obj_init_pos+d_move[0]*v_move, self.obj_low, self.obj_high)
            self.sim.model.body_pos[self.model.body_name2id(
                'faucetBase')] = self.obj_init_pos
            self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125]))
            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d_move[0][0] = -d_move[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d_move[0][1] = -d_move[0][1]
        else:
            exit(0)

        if v_rotate != 0:

            print("rotate")

            self.sim.model.body_quat[self.model.body_name2id('faucetBase')] = random_z_rotation_quaternion(2*math.asin(self.sim.model.body_quat[self.model.body_name2id('faucetBase')][-1])/math.pi*180+v_rotate*d_rotate)
            self.sim.forward()

            self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125]))

            self.sim.data.site_xpos[0] = self._target_pos
            theta = 2*math.asin(self.sim.model.body_quat[self.model.body_name2id('faucetBase')][-1])/math.pi*180

            if theta > 0 or theta < -180:
                d_rotate = -d_rotate
        else:
            exit(0)

        return d_rotate, d_move
           
    def reset_model(self, set_xyz=False):
        self._reset_hand()

        if set_xyz:

            if self.random_init:

                # Compute faucet position
                self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
                    else self.init_config['obj_init_pos']
                # Set mujoco body to computed position
                self.sim.model.body_pos[self.model.body_name2id(
                    'faucetBase')] = self.obj_init_pos
                self.sim.model.body_quat[self.model.body_name2id(
                    'faucetBase')] = random_z_rotation_quaternion()
                self.hand_init_pos = np.random.uniform(low=np.array([-0.3, 0.5, 0.2]), high=np.array([0.3, 0.6, 0.4]))
                self.camera_controller.randomize_camera(method='cylindrical')

                self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125])) #(window_quat, pos, local_offset)

            else:
                self.obj_init_pos = [self.obj_x[self.obj_initial_cnt%5], self.obj_y[self.obj_initial_cnt%4], 0.0]
                self.sim.model.body_pos[self.model.body_name2id(
                    'faucetBase'
                )] = self.obj_init_pos
                self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])
                self.obj_initial_cnt += 1
                self.sim.model.body_quat[self.model.body_name2id('faucetBase')] = random_z_rotation_quaternion(self.obj_theta[self.obj_initial_cnt%20])
                self.sim.forward()
                self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)
                self._target_pos = update_target_pos(self.sim.model.body_quat[self.model.body_name2id('faucetBase')], self.obj_init_pos, np.array([+self._handle_length, .0, .125])) #(window_quat, pos, local_offset)



        else:
            self.sim.model.body_pos[self.model.body_name2id(
                'faucetBase'
            )] = self.obj_init_pos
            self._target_pos = np.array(self._target_pos)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7] + np.array([-.04, .0, .03])
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self._target_radius),
            margin=abs(target_to_obj_init - self._target_radius),
            sigmoid='long_tail',
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid='gaussian',
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place

        reward *= 2

        reward = 10 if target_to_obj <= self._target_radius else reward

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped,
                in_place)
