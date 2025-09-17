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
    
    def euler2quat(self, euler):
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
    
    def mat2quat(self, rotation_matrix):
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


        # if task_type == 'height':
        #     r = math.sqrt(0.6**2+0.295**2)
        #     # z = 0.6
        #     theta = np.pi / 3
        # elif task_type == 'r':
        #     # r = math.sqrt(0.6**2+0.295**2)
        #     z = 0.6
        #     theta = np.pi / 3
        # elif task_type == 'theta':
        #     r = math.sqrt(0.6**2+0.295**2)
        #     z = 0.6
            # theta = np.pi / 3


        r = math.sqrt(0.6**2+0.295**2)
        z = 0.6

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