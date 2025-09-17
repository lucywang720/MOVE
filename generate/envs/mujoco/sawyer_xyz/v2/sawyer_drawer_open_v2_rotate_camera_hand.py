import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from scipy.spatial.transform import Rotation
import random
import math

def update_handle_pos(quat, pos):
    # Step 1: 获取抽屉的当前姿态和位置
    drawer_quat = quat
    drawer_pos = pos
    d = 0.16

    return np.array([drawer_pos[0]+d*math.sin(2*math.asin(drawer_quat[-1])), drawer_pos[1]-d*math.cos(2*math.asin(drawer_quat[-1]))+d, 0.09])
                                                                                             


def random_z_rotation_quaternion(degrees=None):
    """生成随机0-180度绕z轴旋转的纯四元数"""
    if degrees:
        degrees = degrees
    else:
        degrees = random.uniform(0, 90)        # 生成0-180随机角度
    theta = math.radians(degrees)           # 转为弧度
    w = math.cos(theta / 2)                 # 实部计算
    z = math.sin(theta / 2)                 # z虚部分量
    return np.array([w, 0.0, 0.0, z])       # [w, x, y, z]格式

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

task_type = 'theta'

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
                                 theta_range=(0, 2*np.pi), phi_range=(0, np.pi)):
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
                                   height_range=(0.4, 1.0), theta_range=(-np.pi/2, np.pi/2)):
        if center is None:
            center = self.target_point

        print("?")
        exit(0)
        
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



class SawyerDrawerOpenRotateCameraEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)

        obj_low = (-0.3, 0.8, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        self.obj_low = obj_low
        self.obj_high = obj_high

        self.obj_x = [-0.48, -0.43, -0.379, -0.328, -0.278, -0.227, -0.177, -0.126, -0.076, -0.025, 0.025, 0.076, 0.126, 0.177, 0.227, 0.278, 0.328, 0.379, 0.429, 0.48]

        self.obj_x = [-0.28, -0.2, -0.1, 0, 0.08]
        self.obj_y = [0.82, 0.84, 0.86, 0.88]
        self.obj_theta = [2, 5, 10, 15, 20, 25, 30, 35, 40, 42, 45, 48, 50, 55, 65, 70, 75, 80, 85, 88]

        self.hand_x = [-0.28, -0.18, -0.13, -0.08, 0, 0.08, 0.13, 0.18, 0.23, 0.28]
        self.hand_y = [0.52, 0.54, 0.56, 0.58]
        self.hand_z = [0.22, 0.26, 0.3, 0.34, 0.38]

        # self.theta_cam = [np.pi/21*i for i in range(1,21)]
        self.theta_cam = [np.pi/21*i-np.pi/3 for i in range(1,21)]


        self.obj_initial_cnt = 0

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.retarget = False
        self.move_object = False
        self.rotate_object = True

        self.move_step = 45

        self.not_grasp_final = True


        self._target_pos = np.array([0, 0.6, 0.2])

        self.camera_controller = CameraController(self)


        

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    def update_target_pos(self):
        # Step 1: 获取抽屉的当前姿态和位置
        drawer_quat = self.sim.data.get_body_xquat('drawercase_link')  # 四元数格式 (w,x,y,z)
        drawer_pos = self.sim.data.get_body_xpos('drawer')     # 全局位置

        # Step 2: 四元数转旋转矩阵
        quat_scipy = np.roll(drawer_quat, -1)  # 转换为Scipy格式 (x,y,z,w)
        R = Rotation.from_quat(quat_scipy).as_matrix()

        # Step 3: 计算全局偏移量
        # local_offset = np.array([0.0, -0.16, 0.2])
        local_offset = np.array([0.0, -0.16 - self.maxDist, 0.09])
        global_offset = R.dot(local_offset)

        # print(global_offset)
        # exit(0)

        # Step 4: 更新目标位置
        self._target_pos = drawer_pos + global_offset
        d = 0.16 + self.maxDist
        self._target_pos = np.array([drawer_pos[0]+d*math.sin(2*math.asin(drawer_quat[-1])), drawer_pos[1]-d*math.cos(2*math.asin(drawer_quat[-1])), 0.09])


        # print("update target", global_offset, self._target_pos, drawer_pos, self.sim.data.get_body_xpos('drawer_link'), self.sim.data.get_body_xpos('drawer') )
        # exit(0)

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(handle_error <= 0.03),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('drawer_link') + np.array([.0, -.16, .0])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('drawer_link')

    def reset_model(self, set_xyz=False):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        if set_xyz:

            if self.random_init:
                # Compute nightstand position
                self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
                    else self.init_config['obj_init_pos']
                # Set mujoco body to computed position
                self.sim.model.body_pos[self.model.body_name2id(
                    'drawer'
                )] = self.obj_init_pos
                self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion()
                self.sim.forward()
                self.update_target_pos()
                self.hand_init_pos = np.random.uniform(low=np.array([-0.3, 0.5, 0.2]), high=np.array([0.3, 0.6, 0.4]))
                self.camera_controller.randomize_camera(method='cylindrical')

            else:
                self.obj_init_pos = [self.obj_x[self.obj_initial_cnt%5], self.obj_y[self.obj_initial_cnt%4], 0.0]
                self.sim.model.body_pos[self.model.body_name2id(
                    'drawer'
                )] = self.obj_init_pos
                self.obj_initial_cnt += 1
                self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion(self.obj_theta[self.obj_initial_cnt%20])
                self.sim.forward()
                self.update_target_pos()
                self.hand_init_pos = np.array([self.hand_x[self.obj_initial_cnt%10], self.hand_y[self.obj_initial_cnt%4], self.hand_z[self.obj_initial_cnt%5]])
                self.camera_controller.set_cylindrical_camera(radius=math.sqrt(0.6**2+0.295**2), theta=self.theta_cam[self.obj_initial_cnt%20], height=0.6)


            # self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

        else:
            self.sim.model.body_pos[self.model.body_name2id(
                'drawer'
            )] = self.obj_init_pos
            self._target_pos = self._target_pos

            # self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])
            # self.update_target_pos()

        # self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion()
        # self.sim.forward()
        # self.update_target_pos()


        return self._get_obs()

    def set_new_target(self):
        return


    def set_new_obj(self, state=None, v=0, d=None):
        print("!")
        if (v is None or v==0) and d is None:
            print("?")
            self.obj_init_pos = np.array([
                np.random.uniform(self.obj_low[0], self.obj_high[0]),  # 第一个元素：[-0.3, 0.3]
                np.random.uniform(self.obj_low[1], self.obj_high[1]),   # 第二个元素：[0.6, 0.8]
                np.random.uniform(self.obj_low[2], self.obj_high[2])      # 第三个元素：[0, 0.2]
            ])
            self.sim.model.body_pos[self.model.body_name2id(
                'drawer'
            )] = self.obj_init_pos
            self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

            self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion()
            self.sim.forward()
            self.update_target_pos()
            self.sim.data.site_xpos[0] = self._target_pos
            return


        if v != 0:

            self.obj_init_pos = np.clip(self.obj_init_pos+d[0]*v, self.obj_low, self.obj_high)
            self.sim.model.body_pos[self.model.body_name2id(
                'drawer'
            )] = self.obj_init_pos
            # Set _target_pos to current drawer position (closed) minus an offset
            self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d[0][0] = -d[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d[0][1] = -d[0][1]

            return d

        else:
            exit(0)         
 
    def rotate_obj(self, state=None, v=0, d=None):
        if v != 0:

            self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion(2*math.asin(self.sim.model.body_quat[self.model.body_name2id('drawercase_link')][-1])/math.pi*180+v*d)
            self.sim.forward()
            self.update_target_pos()
            self.sim.data.site_xpos[0] = self._target_pos
            theta = 2*math.asin(self.sim.model.body_quat[self.model.body_name2id('drawercase_link')][-1])/math.pi*180

            if theta > 90 or theta < 0:
                d = -d

            return d

        else:
            exit(0)         
 
    def set_rotate_and_new_obj(self, state=None, v_rotate=0, d_rotate=None, v_move=0, d_move=None):
        if v_move != 0:

            self.obj_init_pos = np.clip(self.obj_init_pos+d_move[0]*v_move, self.obj_low, self.obj_high)
            self.sim.model.body_pos[self.model.body_name2id(
                'drawer'
            )] = self.obj_init_pos

            if self.obj_init_pos[0] == self.obj_low[0] or self.obj_init_pos[0] == self.obj_high[0]:
                d_move[0][0] = -d_move[0][0]
            if self.obj_init_pos[1] == self.obj_low[1] or self.obj_init_pos[1] == self.obj_high[1]:
                d_move[0][1] = -d_move[0][1]
        else:
            exit(0)

        if v_rotate != 0:

            self.sim.model.body_quat[self.model.body_name2id('drawercase_link')] = random_z_rotation_quaternion(2*math.asin(self.sim.model.body_quat[self.model.body_name2id('drawercase_link')][-1])/math.pi*180+v_rotate*d_rotate)
            self.sim.forward()

            self.update_target_pos()
            self.sim.data.site_xpos[0] = self._target_pos
            theta = 2*math.asin(self.sim.model.body_quat[self.model.body_name2id('drawercase_link')][-1])/math.pi*180

            if theta > 90 or theta < 0:
                d_rotate = -d_rotate
        else:
            exit(0)

        return d_rotate, d_move


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

    

    # def compute_reward(self, action, obs):
    #     gripper = obs[:3]
    #     handle = obs[4:7]
    #     # handle = update_handle_pos(obs[7:11], obs[4:7])


    #     handle_error = np.linalg.norm(handle - self._target_pos)

    #     print("reward: ", handle, self._target_pos, handle_error)


    #     reward_for_opening = reward_utils.tolerance(
    #         handle_error,
    #         bounds=(0, 0.02),
    #         margin=self.maxDist,
    #         sigmoid='long_tail'
    #     )

    #     handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
    #     # Emphasize XY error so that gripper is able to drop down and cage
    #     # handle without running into it. By doing this, we are assuming
    #     # that the reward in the Z direction is small enough that the agent
    #     # will be willing to explore raising a finger above the handle, hook it,
    #     # and drop back down to re-gain Z reward
    #     scale = np.array([3., 3., 1.])
    #     gripper_error = (handle - gripper) * scale
    #     gripper_error_init = (handle_pos_init - self.init_tcp) * scale

    #     reward_for_caging = reward_utils.tolerance(
    #         np.linalg.norm(gripper_error),
    #         bounds=(0, 0.01),
    #         margin=np.linalg.norm(gripper_error_init),
    #         sigmoid='long_tail'
    #     )

    #     reward = reward_for_caging + reward_for_opening
    #     reward *= 5.0

    #     return (
    #         reward,
    #         np.linalg.norm(handle - gripper),
    #         obs[3],
    #         handle_error,
    #         reward_for_caging,
    #         reward_for_opening
    #     )



    def compute_reward(self, action, obs):
        # 获取观测数据
        gripper_global = obs[:3]      # 夹爪全局坐标 [x, y, z]
        handle_global = update_handle_pos(obs[7:11], obs[4:7])      # 手柄全局坐标 [x, y, z]
        # drawer_pos = self._target_pos # 抽屉的全局目标位置
        drawer_pos = obs[4:7]
        quat = obs[7:11] # 从环境获取抽屉四元数 [x, y, z, w] 格式

        # print(self._target_pos, self._target_pos[-1], self._target_pos[-1] == 0.09)

        handle_pos_init = self._target_pos - np.array([math.sin(2*math.asin(quat[-1])), -math.cos(2*math.asin(quat[-1])), -1]) * self.maxDist

        scale = np.array([3., 3., 1.])
        gripper_error = (handle_global - gripper_global) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        # print("gripper error: ",gripper_error, np.linalg.norm(gripper_error), gripper_error_init, handle_global, gripper_global, handle_pos_init, self.init_tcp)

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        # ---------------------- 手柄移动奖励（全局坐标系） ----------------------
        handle_error_global = np.linalg.norm(handle_global - self._target_pos)
        reward_for_opening = reward_utils.tolerance(
            handle_error_global,
            bounds=(0, 0.02),
            margin=self.maxDist,
            sigmoid="long_tail",
        )

        # ---------------------------- 总奖励计算 ----------------------------
        reward = (reward_for_caging + reward_for_opening) * 5.0

        # print("reward", reward_for_caging, reward_for_opening, reward)

        # 返回调试信息（可选）
        return (
            reward,
            np.linalg.norm(handle_global - gripper_global),  # 全局距离
            obs[3],              # 夹爪状态（如开合程度）
            handle_error_global, # 手柄全局误差
            reward_for_caging,   # 夹爪对准奖励分量
            reward_for_opening,  # 手柄移动奖励分量
        )


if __name__ == '__main__':
    print(compute_ee_quat(np.array([-0.707, 0, 0, 0.707]), offset_rotation=None))