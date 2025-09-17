class CameraController:
    def __init__(self, env, camera_name='corner2', task_name=''):
        self.env = env
        self.sim = env.sim
        self.camera_name = camera_name
        self.target_point = np.array([0.0, 0.6, 0.2])  # 桌子中心点
        self.r, self.theta, self.z = 0, 0, 0
        
        # 获取摄像机ID
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
        
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return quat_wxyz
    
    def euler2quat(self, euler):
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
    
    def mat2quat(self, rotation_matrix):
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