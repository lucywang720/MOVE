import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import time
import os
import matplotlib.pyplot as plt
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            # print("into it")
            n_envs = n_train + n_test
        
        # print(n_envs)

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        save_dir=output_dir,
                        n_action_steps=env_n_action_steps,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', 'train' + str(seed) + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', 'test' + str(seed) + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.seed = seed
        self.output_dir = output_dir
    
    def run(self, policy: BaseLowdimPolicy, verify_policy: BaseLowdimPolicy, threshold):
        device = policy.device
        dtype = policy.dtype
        
        first_n_action = 32
        n_action = 32
        short_T = 36

        Da = policy.action_dim
        del verify_policy
        print('################## Use policy itself as the verifirer ##################')

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            
            # Vars for speculative policy
            old_obs = None
            action_buffer = None
            cur_time = 0
            start = To = self.n_obs_steps
            if policy.oa_step_convention:
                start = To - 1

            long_T = policy.horizon # 132
            assert policy.pred_action_steps_only is False, "Not Implemented for speculative policy"
            chunking_list = []
            infer_count = 0

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                # print(obs.shape) # (bs, 2, 40)
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                if action_buffer == None or cur_time + short_T > long_T:
                    with torch.no_grad():
                        # torch.cuda.synchronize()
                        # cur_time = time.time()
                        
                        action_dict = policy.predict_action(obs_dict)
                        infer_count += policy.num_inference_steps
                        
                        # torch.cuda.synchronize()
                        # predict_time = time.time()-cur_time
                        # print(f'policy {long_T} denoise {policy.num_inference_steps} times with {predict_time} seconds ')
                            
                        # print(type(policy), action_dict['action'].shape, self.n_latency_steps)
                        
                        action_buffer = action_dict
                        old_obs = obs_dict
                        exe_action = action_buffer['action_pred'][:, start:start+first_n_action]
                        cur_time = first_n_action
                        # add a new chunk
                        chunking_list.append(first_n_action)
                        
                else:
                    assert cur_time > 0 and cur_time + short_T <= long_T
                    
                    # The log prob of old actions under old observation
                    
                    # torch.cuda.synchronize()
                    # _time = time.time()
                    
                    # old_log_prob = policy.log_prob(old_obs, action_buffer['naction_pred'], a_t_start=cur_time+start, at_t_end=cur_time+start+n_action)
                    infer_count += 1
                    
                    # torch.cuda.synchronize()
                    # log_prob_time = time.time()-_time
                    # print(f'policy {long_T} estimate log prob with {log_prob_time} seconds ')
                    
                    # The log prob of old actions under new observation
                    new_obs = obs_dict
                    old_naction = action_buffer['naction_pred'][:, cur_time:]
                    
                    # torch.cuda.synchronize()
                    # _time = time.time()
                    
                    with torch.no_grad():
                        new_log_prob_dict = policy.log_prob(new_obs, old_naction, a_t_start=start, at_t_end=start+n_action, )
                    infer_count += 1
                    
                    # torch.cuda.synchronize()
                    # log_prob_time = time.time()-_time
                    # print(f'policy {short_T} estimate log prob with {log_prob_time} seconds ')
                    
                    # v1
                    # ratio = torch.exp(new_log_prob).mean() / torch.exp(old_log_prob).mean()
                    
                    # v2
                    # ratio = torch.exp(new_log_prob.mean() - old_log_prob.mean())
                    # print(f'{new_log_prob.squeeze(-1)=}, {old_log_prob.squeeze(-1)=}')
                    # print(f'{ratio=}')
                    
                    ######## generate actions under new observation for validatation ###############
                    with torch.no_grad():
                        naction_with_new_obs = policy.predict_action(obs_dict, n_repeat=policy.verify_bs, T=short_T)['naction_pred_repeat'].detach()
                    # naction_with_new_obs_long = policy.predict_action(obs_dict, n_repeat=policy.verify_bs)['naction_pred_repeat'].detach()
                    naction_delta = ((naction_with_new_obs[:,:, start:start+n_action] - old_naction[:, start:start+n_action])**2).sum(-1).detach().squeeze(1) # (n_repeat, T)
                    
                    # # create subfigures: one is action dim0; another is action dim1
                    # axes = plt.subplots(2, 1, figsize=(10, 10))[1]
                    
                    # x = np.arange(n_action)
                    # naction_with_new_obs_np = naction_with_new_obs.detach().cpu().numpy()
                    # old_naction_np = old_naction.detach().cpu().numpy()
                    # axes[0].plot(x, old_naction_np[0, start:start+n_action, 0], label="old_naction", color='k')
                    # axes[1].plot(x, old_naction_np[0, start:start+n_action, 1], label="old_naction", color='k')
                    # for i in range(naction_with_new_obs_np.shape[0]):
                    #     axes[0].plot(x, naction_with_new_obs_np[i, 0, start:start+n_action, 0],
                    #                 #  label=f"naction_with_new_obs_{i}",
                    #                  color='r')
                    #     axes[1].plot(x, naction_with_new_obs_np[i, 0, start:start+n_action, 1], 
                    #                 #  label=f"naction_with_new_obs_{i}", 
                    #                  color='r')
                    
                    # # x = np.arange(132-start)
                    # # naction_with_new_obs_long_np = naction_with_new_obs_long.detach().cpu().numpy()
                    # # for i in range(naction_with_new_obs_np.shape[0]):
                    # #     axes[0].plot(x, naction_with_new_obs_long_np[i, 0, start:, 0], color='g')
                    # #     axes[1].plot(x, naction_with_new_obs_long_np[i, 0, start:, 1], color='g')    
                    
                    # # axes[0].legend()
                    # # axes[1].legend()
                    # plt.savefig('x-actions0.jpg')
                    
                    # _naction_with_new_obs = naction_with_new_obs[:,:, start:start+n_action].squeeze(1) # (n_repeat, n_action, Da)
                    # # calculate the dist of one sample in naction_with_new_obs to other samples excluding itself
                    # naction_delta_new = ((_naction_with_new_obs.unsqueeze(1) - _naction_with_new_obs.unsqueeze(0)) **2).sum(-1).mean(-1) # (n_repeat, n_repeat)
                    # # set diag as infinity
                    # naction_delta_new.fill_diagonal_(float('inf'))
                    # naction_delta_new = naction_delta_new.min(-1)[0] # (n_repeat)
                    # _dist_new = naction_delta_new.mean()
                    
                    for method, new_log_prob in new_log_prob_dict.items():
                        estimation = -new_log_prob.squeeze(0)
                        naction_delta_min, _ = naction_delta.min(0)
                        print(naction_delta_min)  
                        print(estimation)           
                        
                        correlation = torch.corrcoef(torch.stack((naction_delta_min, estimation)))[0, 1]
                        m = ((naction_delta_min - naction_delta_min.mean()) * (estimation - estimation.mean())).sum() / ((naction_delta_min - naction_delta_min.mean())**2).sum()
                        b = estimation.mean() - m * naction_delta.mean()

                        print(method, f"Correlation: {correlation.item()}, y = {m.item()}x + {b.item()}")
                        
                        naction_delta_np = naction_delta_min.cpu().numpy()  # Convert to numpy array
                        estimation_np = estimation.cpu().numpy()        # Convert to numpy array
                        
                        plt.scatter(naction_delta_np, estimation_np, alpha=0.7, label=f"{method} (corr: {correlation.item():.2f}, y = {m.item():.2f}x + {b.item():.2f})")
                    plt.legend()
                    plt.savefig('x-y.jpg')
                    plt.close()
                    
                    # # x is time [0, 1, 2, ...]
                    # x = torch.arange(long_T-cur_time, device=naction_delta.device).float()
                    # correlation = torch.corrcoef(torch.stack((x, naction_delta_long)))[0, 1]
                    # plt.scatter(x.cpu().numpy(), naction_delta_long.cpu().numpy(), alpha=0.7, label=f"(corr: {correlation.item():.2f})")
                    # # plot x v.s. norm of naction
                    # norm  = torch.norm(old_naction[:, :long_T-cur_time], dim=-1).squeeze(0).cpu().numpy()
                    # plt.scatter(x.cpu().numpy(), norm, alpha=0.7, label=f"norm of naction")

                    # plt.legend()
                    # plt.savefig('xtime-y.jpg')
                    # plt.close()
                    
                    # _dist = naction_delta.mean(-1).min(0)[0]
                    
                    
                    # find the longest index i that is less than threshold for naction_delta[:i].mean(-1).min(0)[0]
                    longest_index = -1
                    _threshold = Da * threshold**2

                    cumsums = naction_delta.cumsum(dim=1)  # shape (M, N)
                    M, N = naction_delta.shape
                    divisors = torch.arange(1, N + 1, device=naction_delta.device).float()  # shape (N,)
                    row_means_2d = cumsums / divisors  # broadcasting to shape (M, N)
                    row_mins, _ = row_means_2d.min(axis=0)  # shape (N,)
                    delta_mins, _ = naction_delta.min(axis=0)  # shape (N,)
                    valid_indices = torch.nonzero((row_mins < _threshold) & (delta_mins < 2.0 * _threshold)).squeeze(-1)
                    if len(valid_indices) == 0:
                        # No index satisfies the condition
                        largest_i = 0
                    else:
                        # +1 because k is 0-based index, but subarray size is (k+1)
                        largest_i = valid_indices[-1].item() + 1
                    
                    # print(_dist, _dist_new)
                    # ratio = 0
                    
                    ################################################################################
                    
                    if largest_i >= 4:
                        print(f'use old actions, {cur_time=}')
                        exe_action = action_buffer['action_pred'][:, start+cur_time:start+cur_time+largest_i]
                        cur_time += largest_i
                        # update current chunking size
                        chunking_list[-1] += largest_i
                    else:
                        
                        with torch.no_grad():
                            # torch.cuda.synchronize()
                            # cur_time = time.time()
                            
                            action_dict = policy.predict_action(obs_dict)
                            infer_count += policy.num_inference_steps
                            
                            # torch.cuda.synchronize()
                            # predict_time = time.time()-cur_time
                            # print(f'policy {long_T} denoise {policy.num_inference_steps} times with {predict_time} seconds ')
                            
                            # print(type(policy), action_dict['action'].shape, self.n_latency_steps)
                            action_buffer = action_dict
                            old_obs = obs_dict
                        exe_action = action_buffer['action_pred'][:, start:start+first_n_action]
                        cur_time = first_n_action
                        # add a new chunk
                        chunking_list.append(first_n_action)
                        
                
                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                # action = np_action_dict['action'][:,self.n_latency_steps:]


                file_path = os.path.join(self.output_dir, 'media', 'chunking')
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(self.output_dir+f'/media/chunking/{self.env_prefixs[0][:-1]}_{self.seed}.txt', 'w') as f:
                    f.write(str(chunking_list))
                with open(self.output_dir+'/chunking_list.txt', 'w') as f:
                    f.write(str(chunking_list))

                # step env
                exe_action = exe_action.detach().to('cpu').numpy()
                obs, reward, done, info = env.step(exe_action)
                # print(action.shape) (56, 64, 2)
                # print(obs.shape) (56, 2, 40)
                # print(reward.shape) (56,)
                # print(done.shape) (56,)
                
                done = np.all(done)
                past_action = exe_action

                # update pbar
                pbar.update(exe_action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            print(infer_count, chunking_list)
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix+f'chunk_{seed}'] = chunking_list
            log_data[prefix+f'infer_count_{seed}'] = infer_count
            log_data[prefix+f'avg_chunk_{seed}'] = np.mean(chunking_list)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
