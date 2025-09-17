import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time
from scipy.stats import truncnorm

from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class MetaWorldMoveEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 retarget_low=10, retarget_high=80, max_retargets=2, sampling_type='uniform', velocity=0.0, velocity_camera=0.0, velocity_obj=0.0
                 ):
        super(MetaWorldMoveEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.task_name = task_name

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        # self.env.action_scale = 0.02 / 2
        # print(type(self.env), vars(self.env))
        # exit(0)
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.015, 0.7]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = 128
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points # 512
        
        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
    
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        self.reset_num = 0
        self.success_reward_step = 0
        self.all_trajectory_step = 0

        self.retarget_high = retarget_high
        self.retarget_low = retarget_low
        print(self.retarget_high, self.retarget_low)
        self.retargeted = False
        self.already_change = False

        self.max_retargets = max_retargets
        self.sampling_type = sampling_type

        self.reset_obj_num = 0 #3
        self.reset_obj_cnt = 0
        self.grasp_cnt = 0
        self.retarget_step_list = []

        self.velocity = velocity
        self.velocity_camera = velocity_camera
        self.velocity_obj = velocity_obj

        v = np.random.beta(a=2, b=5, size=1)[0]

        self.v = v * self.velocity
        # self.v = 0
        self.d = np.hstack([np.random.uniform(-1, 1, (1, 2)), np.zeros((1, 1))])

        if self.velocity:
            self.v_theta_camera = self.v / self.velocity * self.velocity_camera / 180 * np.pi * np.random.choice([-1,1])
        elif self.velocity_camera:
            self.v_theta_camera = np.random.beta(a=2, b=5, size=1)[0] * self.velocity_camera / 180 * np.pi * np.random.choice([-1,1])
        else:
            self.v_theta_camera = 0

        self.reset_target = np.random.randint(0, 20)

        self.v_obj = v * self.velocity_obj


        
    
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1]
        
        return point_cloud, depth
        

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict

    def reset_sampling(self, sampling='uniform', low=0, high=0):
        lower, upper = low, high  # 采样区间
        n_samples = 1       # 采样数量

        if sampling == 'uniform':
            sample = np.random.uniform(low=lower, high=upper, size=n_samples)
        elif sampling == 'beta':
            beta_a, beta_b = 2, 5  # 形状参数，a,b>0
            sample = lower + (upper - lower) * np.random.beta(beta_a, beta_b, n_samples)
        elif sampling == 'gaussian':
            gauss_mu = (lower + upper) / 2    # 均值设为区间中点
            gauss_sigma = 12                  # 标准差控制数据集中程度
            a_trunc = (lower - gauss_mu) / gauss_sigma
            b_trunc = (upper - gauss_mu) / gauss_sigma
            sample = truncnorm.rvs(
                a_trunc, b_trunc, 
                loc=gauss_mu, 
                scale=gauss_sigma, 
                size=n_samples
            )
        elif sampling == 'const':
            sample = [50]

        return round(sample[0])



    def step(self, action: np.array, collect=False, move_collect=False):
        # print(self.env.data.qpos)
        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1  # 更新总步数计数器

        env_info['set_obj'] = False

        # print('?', env_info['grasp_success'], move_collect)

        # if self.cur_step < 35 and move_collect:
        # print(self.v)

        # for box-close i change 35 to self.env.move_step, formally all 35
        if self.v and self.cur_step < self.env.move_step and move_collect:
            if self.env.retarget and env_info['grasp_success']:
                tmp_d = self.env.set_new_target(v=self.v, d=self.d)
                # print("tmp_d", tmp_d)
                if not (tmp_d is None):
                    self.d = tmp_d

            if self.env.early_not_move and self.cur_step < 10:
                pass
            elif self.v and self.env.move_object:
                self.d = self.env.set_new_obj(v=self.v, d=self.d) #self.d_tmp
                # self.d = self.env.set_new_obj(v=self.v, d=self.d_tmp) #self.d_tmp

        # elif self.cur_step == self.reset_obj_step and self.v:   # used for pick and place move target
        #     self.env.set_new_target()
        # elif self.cur_step == self.reset_obj_step and self.v_obj:
        elif self.v_obj and self.cur_step < self.env.retarget_max_step: #90 -> self.env.retarget_max_step
            # print(self.v_obj, self.d_obj)
            # exit(0)
            self.d_obj = self.env.set_new_target(v=self.v_obj, d=self.d_obj)

        # print(self.v_obj, self.d_obj)
        # exit(0)

        if self.v_theta_camera:
            # print(self.v_theta_camera, "?")
            # exit(0)
            self.v_theta_camera = self.env.set_camera_pos(self.v_theta_camera)


     

        # ===================== 观测数据生成 =====================
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        # ===================== 终止条件判断 =====================
        # 基础终止条件：达到最大步数
        done = done or self.cur_step >= self.episode_length

        # print(reward, env_info['success'])

        if 'drawer' in self.task_name or 'box-close' in self.task_name or 'window-open' in self.task_name:
            if env_info['success'] and collect and self.remaining_retargets==0 and (not self.waiting_for_retarget) and (self.reset_obj_cnt == self.reset_obj_num or not move_collect):
                self.success_reward_step += 1

            if self.success_reward_step > 9:
                done = True
        
        if reward == 10 and collect and self.remaining_retargets==0 and (not self.waiting_for_retarget) and (self.reset_obj_cnt == self.reset_obj_num or not move_collect):
            self.success_reward_step += 1

            if self.success_reward_step > 9:
                done = True

        return obs_dict, reward, done, env_info


    def reset(self):
        self.env.reset()
        print('3285674989tq0')
        self.env.reset_model(set_xyz=True)
        # self.env.reset_model()
        raw_obs = self.env.reset()
        
        # 重置所有状态变量
        self.cur_step = 0
        self.remaining_retargets = self.max_retargets  # 从超参数初始化
        self.target_step = None
        self.waiting_for_retarget = False
        self.success_counter = 0
        self.success_reward_step = 0
        self.reset_obj_step = self.reset_sampling(self.sampling_type, 50, 70)
        self.retarget_step_list = []
        # self.reset_obj_step = 50
        self.reset_obj_cnt = 0
        self.grasp_cnt = 0

        v = np.random.beta(a=2, b=5, size=1)[0]

        self.v = v * self.velocity
        self.d = np.hstack([np.random.uniform(-1, 1, (1, 2)), np.zeros((1, 1))])

        self.v_obj = v * self.velocity_obj
        self.d_obj = np.hstack([np.random.uniform(-1, 1, (1, 2)), np.zeros((1, 1))])

        self.d_tmp = np.array([[1,1,0]])

        if self.velocity:
            self.v_theta_camera = self.v / self.velocity * self.velocity_camera / 180 * np.pi * np.random.choice([-1,1])
        elif self.velocity_camera:
            self.v_theta_camera = np.random.beta(a=2, b=5, size=1)[0] * self.velocity_camera / 180 * np.pi * np.random.choice([-1,1])
        else:
            self.v_theta_camera = 0

        self.v_obj = v * self.velocity_obj

        print(self.reset_obj_step)

        # 观测数据生成（保持原有逻辑）
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }


        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass

