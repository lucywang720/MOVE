import numpy as np
from gym.spaces import Box
import random
import math
import random
from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
import mujoco
from scipy.spatial.transform import Rotation as R

task_type = 'theta'

def random_z_rotation_quaternion(degrees=None):
    """生成随机0-180度绕z轴旋转的纯四元数"""
    # print("!", degrees)
    if degrees:
        degrees = degrees
    else:
        degrees = random.uniform(-60, 60)        # 生成0-180随机角度
    # print("!", degrees)
    
    theta = math.radians(degrees)           # 转为弧度
    w = math.cos(theta / 2)                 # 实部计算
    z = math.sin(theta / 2)                 # z虚部分量
    return np.array([w, 0.0, 0.0, z])       # [w, x, y, z]格式

class CameraController:
    def __init__(self, env, camera_name='corner2', task_name=''):
        """
        初始化摄像机控制器
        
        Args:
            env: MetaWorld环境实例
            camera_name: 摄像机名称，MetaWorld中常用的有 'corner', 'corner2', 'topview' 等
        """
        self.env = env
        self.sim = env.sim
        self.camera_name = camera_name
        self.target_point = np.array([0.0, 0.6, 0.2])  # 桌子中心点
        self.r, self.theta, self.z = 0, 0, 0
        
        # 获取摄像机ID
        try:
            self.camera_id = self.sim.model.camera_name2id(camera_name)
        except:
            print(f"警告: 找不到摄像机 '{camera_name}', 使用默认ID 0")
            self.camera_id = 0
    
    def look_at_euler(self, camera_pos, target_pos):
        """
        使用欧拉角方法计算摄像机朝向
        这种方法更稳定，避免了向量平行的问题
        
        Args:
            camera_pos: 摄像机位置 [x, y, z]
            target_pos: 目标点位置 [x, y, z]
        
        Returns:
            euler_angles: [roll, pitch, yaw] 弧度
        """
        # 计算从摄像机到目标的向量
        dx = target_pos[0] - camera_pos[0]
        dy = target_pos[1] - camera_pos[1]
        dz = target_pos[2] - camera_pos[2]
        
        # 计算水平距离
        horizontal_dist = np.sqrt(dx*dx + dy*dy)
        
        # 计算yaw角 (绕Z轴旋转)
        yaw = np.arctan2(dy, dx)
        
        # 计算pitch角 (绕Y轴旋转) 
        pitch = np.arctan2(dz, horizontal_dist)  # 负号因为MuJoCo坐标系
        
        # roll角设为0 (不倾斜)
        roll = 0

        # print(yaw, pitch, roll)
        # yaw, pitch, roll = 3.9, 2.3, 0.6
        
        return np.array([roll, pitch, yaw])

    def look_at_matrix(self, camera_pos, target_pos, up=np.array([0, 0, -1])):
        eye = np.array(camera_pos, dtype=float)
        target = np.array(target_pos, dtype=float)
        up = np.array(up, dtype=float)
        
        # 计算forward向量（从摄像机指向目标）
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        
        # 计算right向量 
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        
        if right_norm < 1e-6:
            # 处理forward与up平行的情况
            if abs(forward[2]) > 0.9:
                up = np.array([1, 0, 0])
            else:
                up = np.array([0, 0, 1])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
        
        right = right / right_norm
        
        # 重新计算up向量确保正交
        up_corrected = np.cross(right, forward)
        
        # 构建旋转矩阵 (MuJoCo坐标系：X右，Y上，Z后)
        R_matrix = np.array([
            right,           # X轴
            up_corrected,    # Y轴
            -forward         # Z轴（摄像机看向-Z）
        ]).T
        
        # 转换为四元数
        from scipy.spatial.transform import Rotation as R_scipy
        r = R_scipy.from_matrix(R_matrix)
        quat_xyzw = r.as_quat()
        
        # 转换为MuJoCo的wxyz格式
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return quat_wxyz
    
    def euler_to_quaternion(self, euler):
        """
        将欧拉角转换为四元数 (MuJoCo wxyz格式)
        """
        roll, pitch, yaw = euler
        
        # 使用ZYX顺序 (yaw, pitch, roll)
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
        quat_xyzw = r.as_quat()  # scipy返回xyzw顺序
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return quat_wxyz
    
    def set_camera_pose(self, position, quaternion=None):
        position = np.array(position, dtype=float)
        
        # 检查位置是否合理
        dist_to_target = np.linalg.norm(position - self.target_point)
        if dist_to_target < 0.1:
            return False
        
        # 检查是否在地面以下
        if position[2] < 0:
            position[2] = 0.1  # 修正到地面上方
        
        if quaternion is None:
            # 使用新的欧拉角方法计算朝向
            euler = self.look_at_euler(position, self.target_point)
            # quaternion = self.euler_to_quaternion(euler)
            quaternion = self.look_at_matrix(position, self.target_point)
        
        # 验证四元数
        quat_norm = np.linalg.norm(quaternion)
        if abs(quat_norm - 1.0) > 1e-3:
            print(f"警告: 四元数不是单位四元数，norm={quat_norm:.3f}")
            quaternion = quaternion / quat_norm
        
        # 设置摄像机位置和姿态
        self.env.model.cam_pos[self.camera_id] = position
        self.env.model.cam_quat[self.camera_id] = quaternion

        # print(position, quaternion)
        
        # 更新模拟
        # self.sim.forward()
        
        # print(f"摄像机设置: 位置={position}, 距离目标={dist_to_target:.3f}")
        return True

    def sample_spherical_position(self, center=None, radius_range=(0.8, 1.5), 
                                 theta_range=(0, 2*np.pi), phi_range=(np.pi/6, 2*np.pi/3)):
        if center is None:
            center = self.target_point
        
        # 随机采样球坐标参数
        r = np.random.uniform(radius_range[0], radius_range[1])
        theta = np.random.uniform(theta_range[0], theta_range[1])
        phi = np.random.uniform(phi_range[0], phi_range[1])
        
        # 转换为笛卡尔坐标
        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)
        
        # 确保摄像机不会在地面以下
        z = max(z, 0.1)
        
        return np.array([x, y, z])
    
    def sample_cylindrical_position(self, center=None, radius_range=(0.8, 1.5), 
                                   height_range=(0.4, 1.0), theta_range=(0, np.pi)):
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
        
        # 转换为笛卡尔坐标
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        z = center[2] + z
        
        # 确保摄像机不会在地面以下
        z = max(z, 0.1)
        
        return np.array([x, y, z])
    
    def sample_preset_positions(self):
        """
        从预设的几个好的观察位置中随机选择
        
        Returns:
            position: 预设位置之一
        """
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
        """
        随机化摄像机位置
        
        Args:
            method: 采样方法，'spherical', 'cylindrical', 或 'preset'
            **kwargs: 传递给对应采样方法的参数
        """
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
        """
        获取当前摄像机信息
        """
        pos = self.sim.model.cam_pos[self.camera_id].copy()
        quat = self.sim.model.cam_quat[self.camera_id].copy()
        
        return {
            'position': pos,
            'quaternion': quat,
            'target': self.target_point,
            'distance_to_target': np.linalg.norm(pos - self.target_point)
        }



class SawyerNutDisassembleRotateAndBothCameraHandEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02
    delta_xyz = np.array([0, 0, 0])

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        # obj_low = (0.1, 0.6, 0.025)
        # obj_high = (0.1, 0.75, 0.02501)
        obj_low = (-0.35, 0.5, 0.025)
        obj_high = (0.1, 0.7, 0.02501)

        goal_low = (-0.35, 0.7, 0.1699)
        goal_high = (0.1, 0.7, 0.1701)
        


        self.goal_low = goal_low
        self.goal_high = goal_high
        self.obj_low = obj_low
        self.obj_high = obj_high


        self.obj_x = [-0.32, -0.21, -0.1, 0.01, 0.13]
        self.obj_y = [0.52, 0.57, 0.62, 0.68]

        self.rotate = [-55, -50, -45, -40, -35, -30, -25, -15, -10, -5, 1e-10, 10, 15, 25, 30, 35, 40, 45, 50, 55]

        self.grasp_final = False


        # 添加网格模式数组定义
        self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]

        # radius_range=(0.8, 1.5), height_range=(0.4, 1.0), theta_range=(0, np.pi))
        self.r_cam = [0.82, 0.86, 0.9, 0.94, 0.98, 1.02, 1.06, 1.1, 1.12, 1.14, 1.16, 1.18, 1.22, 1.26, 1.3, 1.34, 1.38, 1.42, 1.45, 1.48]
        self.h_cam = [0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.6, 0.63, 0.66, 0.69, 0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.9, 0.93, 0.96, 0.98]
        self.theta_cam = [np.pi/21*i for i in range(1,21)]

        self.obj_initial_cnt = 0

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            # obj_hand_rotate=True   ???需要吗
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.7, 0.025]),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        # self.retarget = False
        # self.move_object = False
        # self.rotate_object = True

        self.move_step = 20

        self._target_pos = self.goal.copy()

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([.0, .0, .005]),
            np.array(goal_high) + np.array([.0, .0, .005])
        )


        self.camera_controller = CameraController(self)
        



    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_assembly_peg.xml')

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
        return [('pegTop', self._target_pos)]

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('WrenchHandle')

    def _get_pos_objects(self):
        return self._get_site_pos('RoundNut-8')

    def _get_quat_objects(self):
        return np.concatenate([self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')], np.concatenate([SawyerNutDisassembleRotateAndBothCameraHandEnvV2.delta_xyz, [0]],axis=0)], axis=0)

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('RoundNut')
        return obs_dict

    def reset_model(self, set_xyz=False):
        self._reset_hand()

        if set_xyz:
            self._target_pos = self.goal.copy()
            self.obj_init_angle = self.init_config['obj_init_angle']

            if self.random_init:
                goal_pos = self._get_state_rand_vec()
                while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                    goal_pos = self._get_state_rand_vec()
                self.obj_init_pos = goal_pos[:3]
                self._target_pos = goal_pos[:3] + np.array([0, 0, 0.15])
                self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')] = random_z_rotation_quaternion()
                self.camera_controller.randomize_camera(method='cylindrical')
                self.hand_init_pos = np.random.uniform(low=np.array([-0.3, 0.5, 0.2]), high=np.array([0.3, 0.6, 0.4]))
            else:
                self.obj_init_pos = np.array([self.obj_x[self.obj_initial_cnt%5], self.obj_y[self.obj_initial_cnt%4], 0.025])
                self._target_pos = self.obj_init_pos + np.array([0, 0, 0.15]) 
                self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')] = random_z_rotation_quaternion(degrees=self.rotate[self.obj_initial_cnt%20])
                self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])

                if task_type == 'height':
                    self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=np.pi/3, height=self.h_cam[self.obj_initial_cnt%20])
                elif task_type == 'r':
                    self.camera_controller.set_cylindrical_camera(radius=self.r_cam[self.obj_initial_cnt%20], theta=np.pi/3, height=0.6)
                elif task_type == 'theta':
                    self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)
                elif task_type == 'all':
                    self.camera_controller.set_cylindrical_camera(radius=self.r_cam[self.obj_initial_cnt%20], theta=self.theta_cam[self.obj_initial_cnt%20], height=self.h_cam[self.obj_initial_cnt%20])
                
                self.obj_initial_cnt += 1

        else:
            self._target_pos = self._target_pos
            self.obj_init_angle = self.init_config['obj_init_angle']



        peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
        self._set_obj_xyz(self.obj_init_pos)
        
        # 同步物理状态并计算偏移量
        self.sim.forward()
        SawyerNutDisassembleRotateAndBothCameraHandEnvV2.delta_xyz = self.data.site_xpos[self.model.site_name2id('RoundNut-8')] - self.data.site_xpos[self.model.site_name2id('RoundNut')]

        return self._get_obs()

    # def reset_model(self):
    #     self._reset_hand()
    #     self._target_pos = self.goal.copy()
    #     self.obj_init_pos = np.array(self.init_config['obj_init_pos'])
    #     self.obj_init_angle = self.init_config['obj_init_angle']

    #     if self.random_init:
    #         goal_pos = self._get_state_rand_vec()
    #         while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
    #             goal_pos = self._get_state_rand_vec()
    #         self.obj_init_pos = goal_pos[:3]
    #         self._target_pos = goal_pos[:3] + np.array([0, 0, 0.15])

    #     peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
    #     peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
    #     self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
    #     self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
    #     self._set_obj_xyz(self.obj_init_pos)

    #     return self._get_obs()


    def rotate_obj(self, state=None, v=0, d=None):
        if v != 0:

            self._set_obj_xyz(self.obj_init_pos)
            self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')] = random_z_rotation_quaternion(2*math.asin(self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')][-1])/math.pi*180+v*d)
            # self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')] = random_z_rotation_quaternion()

            # print(self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')])
            self.sim.forward()
            SawyerNutDisassembleRotateAndBothCameraHandEnvV2.delta_xyz = self.data.site_xpos[self.model.site_name2id('RoundNut-8')] - self.data.site_xpos[self.model.site_name2id('RoundNut')]
            theta = 2*math.asin(self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')][-1])/math.pi*180

            # print(self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')])

            if theta < -60 or theta > 60:
                d = -d

            return d

        else:
            exit(0)   

    def set_new_obj(self, v=None, d=None):
        # print(1)

        if v is None and d is None:
            # print(2)
            self.obj_init_pos = np.array([
                np.random.uniform(self.obj_low[0], self.obj_high[0]),  # 第一个元素：[-0.3, 0.3]
                np.random.uniform(self.obj_low[1], self.obj_high[1]),   # 第二个元素：[0.6, 0.8]
                np.random.uniform(self.obj_low[2], self.obj_high[2])      # 第三个元素：[0, 0.2]
            ])
            self._set_obj_xyz_move(self.obj_init_pos)
            
            # 更新peg位置跟随物体
            peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
            peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
            self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
            self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
            
            self.sim.model.body_quat[self.model.geom_name2id('WrenchHandle')] = random_z_rotation_quaternion()
            self.sim.forward()
            SawyerNutDisassembleRotateAndBothCameraHandEnvV2.delta_xyz = self.data.site_xpos[self.model.site_name2id('RoundNut-8')] - self.data.site_xpos[self.model.site_name2id('RoundNut')]

            return

        if v != 0:
            self.obj_init_pos = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)
            # print("target, ", self._target_pos)
            self._set_obj_xyz_move(self.obj_init_pos)
            
            # 更新peg位置跟随物体
            peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
            peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
            self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
            self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
            
            # print(d)
            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]

            return d


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
            exit(0)  

        return d_rotate, d_move    


    def set_camera_pos(self, v_theta_cam=0, v_r_cam=0, v_h_cam=0):

        theta = self.camera_controller.theta
        r = self.camera_controller.r
        h = self.camera_controller.z
        if task_type == 'theta':
            theta = np.clip(theta+v_theta_cam, 0, np.pi)
        elif task_type == 'r':
            r = np.clip(self.camera_controller.r+v_r_cam, 0.8, 1.5)
        elif task_type == 'height':
            h = np.clip(self.camera_controller.z+v_h_cam, 0.4, 1.0)
        elif task_type == 'all':
            theta = np.clip(self.camera_controller.theta+v_theta_cam, 0, np.pi)
            r = np.clip(self.camera_controller.r+v_r_cam, 0.8, 1.5)
            h = np.clip(self.camera_controller.z+v_h_cam, 0.4, 1.0)

        self.camera_controller.set_cylindrical_camera(radius=r, theta=theta, height=h)
        if r == 0.8 or r == 1.5:
            v_r_cam = -v_r_cam
        if theta == 0 or theta == np.pi:
            v_theta_cam = -v_theta_cam
        if h == 0.4 or h == 1.0:
            v_h_cam = -v_h_cam

        return v_theta_cam, v_r_cam, v_h_cam





    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error / 0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos + np.array([.0, .0, .1]) - wrench_center

        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error),
            bounds=(0, 0.02),
            margin=0.2,
            sigmoid='long_tail',
        )

        return in_place

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos('RoundNut')
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = SawyerNutDisassembleRotateAndBothCameraHandEnvV2.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = SawyerNutDisassembleRotateAndBothCameraHandEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions, wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            high_density=True,
        )
        reward_in_place = SawyerNutDisassembleRotateAndBothCameraHandEnvV2._reward_pos(
            wrench_center,
            self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success
        success = obs[6] > self._target_pos[2]
        if success:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )
