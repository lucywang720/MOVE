import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy.env.metaworld import MetaWorldEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.simple_video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import diffusion_policy.common.logger_util as logger_util
from termcolor import cprint
import cv2
import os
import random
import string

class MetaworldRunner(BaseImageRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512
                 ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.task_name = task_name
        # self.task_name = 'pick-place'


        def env_fn(task_name):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name,device=device, 
                                 use_point_crop=use_point_crop, num_points=num_points)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        # self.eval_episodes = 20
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

        self.num_record = 0

    def run(self, policy: BaseImagePolicy, save_video=True, ema=False):
        # print(vars(self.env.env.env.env))
        # print("obj initial cnt", self.env.env.env.env.obj_initial_cnt)
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        log_data = dict()
        
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            # start rollout
            obs = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                with torch.no_grad():
                    obs_dict_input = {}
                    # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['image'] = obs_dict['image'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, info = env.step(action)

                # print(action.shape, obs.shape)
                # print(action.shape, obs['image'].shape, obs['full_state'].shape, obs['agent_pos'].shape)


                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            # print(is_success)

            videos = env.env.get_video()
            # print("???", videos.shape)
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            
            if save_video or True:
                # videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
                # log_data[f'sim_video_eval'] = videos_wandb

                # 使用 opencv 保存视频
                self.num_record += 1
                _, height, width = videos[0].shape
                # print(videos[0].shape, len(videos))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
                # video_path = f'/diffusion_policy/stick_output_{self.num_record}.mp4'
                letters = string.ascii_letters
                tmp_code = ''.join(random.choice(letters) for _ in range(6))
                video_path = self.output_dir + f'/{self.num_record}_success{is_success}_{tmp_code}.mp4'
                # print(video_path)
                out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

                for frame in videos:
                    out.write(frame.transpose(1, 2, 0))
                out.release()

                # 记录视频路径到 W&B
                log_data['sim_video_eval_path'] = video_path

                all_success_rates.append(is_success)
                all_traj_rewards.append(traj_reward)
                

        max_rewards = collections.defaultdict(list)

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        if ema:
            cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'red')
        else:
            cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        with open(self.output_dir + '/sr.txt', 'a') as f:
            f.write(str(np.mean(all_success_rates)) + '\n')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        _ = env.reset()
        videos = None

        # print("obj initial cnt", self.env.obj_initial_cnt)
        # self.env.obj_initial_cnt = 0
        # print("obj initial cnt", self.env.obj_initial_cnt)
        self.env.env.env.env.obj_initial_cnt = 0


        return log_data
