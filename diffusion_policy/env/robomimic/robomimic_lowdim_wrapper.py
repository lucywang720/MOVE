from typing import List, Dict, Optional
import numpy as np
import gym
import cv2
import os
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite
from robosuite.models.base import MujocoModel

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview',
        n_action_steps=128,
        steps_per_render=2,
        save_dir=None
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        self.save_dir = save_dir
        
        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        self.path_coords = []
        self.n_action_steps = n_action_steps // steps_per_render
        # print(n_action_steps, steps_per_render)
        self.now_step = 0
        # self.steps_per_render = steps_per_render
        self.color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (255,0,255), (0, 0, 0), (69, 143, 153), (87, 224, 184), (11, 164, 203), (117, 177, 207), (197, 69, 16), (128, 0, 128), (0, 128, 128), (128, 64, 0), (64, 0, 128), (0, 128, 64), (255, 128, 0), (64, 128, 0), (128, 0, 64)]
        self.render_step_list = []

    def get_observation(self):
        raw_obs = self.env.get_observation()
        # print("raw obs shape", raw_obs['robot0_eef_pos'], raw_obs['robot0_eef_quat'], raw_obs['robot0_gripper_qpos'])
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        self.path_coords = []
        self.now_step = 0
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info

    def world_to_pixel(self, xyz, camera_name, env):
        # 获取相机信息
        cam_id = env.env.sim.model.camera_name2id(camera_name)
        
        # 获取相机位置和旋转矩阵
        import copy
        cam_pos = copy.deepcopy(env.env.sim.data.get_camera_xpos(camera_name))
        # cam_pos[-1] = -cam_pos[-1]
        # print(dir(env.env.sim.data))
        cam_mat = env.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)
        

        # print("Camera Position:", cam_pos)
        # print("Camera Matrix:\n", cam_mat)
        # print("xyz:", xyz)
        # 获取相机参数
        fovy = env.env.sim.model.cam_fovy[cam_id]
        # f = 1.0 / np.tan(fovy / 2)
        f = 1.0 / np.tan(fovy / 360.0 * np.pi)
        # print("fovy", fovy, f)
        
        height, width = self.render_hw
        
        # 将世界坐标转换到相机坐标
        # cam_xyz = (xyz - cam_pos).dot(cam_mat.T)
        cam_xyz = (xyz - cam_pos).dot(cam_mat)
        # print("cam_xyz:", cam_xyz)
        
        # 透视投影
        x, y, z = cam_xyz
        # x, y, z = cam_xyz
        # z = -z  # Invert z-coordinate
        # if z <= 0:
            # print("???????? z >= 0")
        px = x / (-z) * f
        py = y / (-z) * f       
        
        # 转换到像素坐标
        u = int((px + 1) * width / 2)
        v = int((py + 1) * height / 2)

        u = max(0, min(u, width - 1))
        v = max(0, min(v, height - 1))
        
        return u, v


        
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        self.env.env.visualize(vis_settings={'env': False, 'grippers': True, 'robots': False})

        # print("---",  self.now_step, self.render_camera_name) #, self.env.env.bottom_offset())

        if not self.save_dir:
            self.save_dir = ''
        elif os.path.isfile(self.save_dir+'/chunking_list.txt'):
            with open(self.save_dir+'/chunking_list.txt', 'r') as f:
                self.render_step_list = eval(f.read())

        u, v = self.world_to_pixel(self.env.env.sim.data.get_site_xpos(self.env.env.robots[0].gripper.important_sites["grip_site"]), self.render_camera_name, self.env)
        # self.env.env.sim.data.set_joint_qpos("object_joint", [1,1,1,0.1,0.1,0.1,0.1])
        # 修改方形坚果位置
        # print(self.env.env.sim.data.get_joint_qpos("SquareNut_joint0"))
        import random
        # self.env.env.sim.data.set_site_xpos("123123123", [ 2.31846784e-01, 1.03699903e-01, 8.29978946e-01, 7.16709051e-01, -5.87630737e-07, 1.11862294e-06, 6.97372308e-01])
        if random.randint(0,1) == 1:
            self.env.env.sim.data.set_joint_qpos("SquareNut_joint0", [ 2.31846784e-01, 1.03699903e-01, 8.29978946e-01, 7.16709051e-01, -5.87630737e-07, 1.11862294e-06, 6.97372308e-01])
        else:
            self.env.env.sim.data.set_joint_qpos("SquareNut_joint0", [ 0.21989982, 0.10698657, 0.95817029, 0.68479111, -0.02860816, -0.00611091, 0.72815202])

        # 修改圆形坚果位置
        # self.env.env.sim.data.set_joint_qpos("RoundNut_joint0", [1,1,1,0.1,0.1,0.1,0.1])
        self.env.env.sim.forward()
        self.now_step += 1
        self.path_coords.append((u,h-v))
        img = self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)

        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        render_list_cnt = 0
        for i in range(1, len(self.path_coords)):
            if self.render_step_list:
                color_index = render_list_cnt
                if i >= sum(self.render_step_list[:render_list_cnt+1]):
                    render_list_cnt += 1
                cv2.line(img_cv, self.path_coords[i-1], self.path_coords[i], self.color[color_index], 1)
            else:
                # print(i, self.n_action_steps)
                cv2.line(img_cv, self.path_coords[i-1], self.path_coords[i], self.color[min(int(i//self.n_action_steps), len(self.color)-1)], 1)

        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        

def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
