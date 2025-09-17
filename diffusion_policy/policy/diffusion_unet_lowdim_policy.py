from typing import Dict
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs): # parameters passed to step
        super().__init__()
        print("into init action", n_action_steps, horizon, n_obs_steps, num_inference_steps, action_dim)
        # n_action_steps = 128
        # n_action_steps = 1
        # horizon = n_action_steps + n_obs_steps + 6
        # horizon = 132
        # horizon = 160
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # Initialize DDIMScheduler with the parameters from noise_scheduler
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler.num_train_timesteps,
            beta_start=noise_scheduler.beta_start,
            beta_end=noise_scheduler.beta_end,
            beta_schedule=noise_scheduler.beta_schedule,
            steps_offset=0,
            clip_sample=True,
            set_alpha_to_one=False
        )
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator) # [56, 132, 2]
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        
        # print(list(scheduler.timesteps)) # [99, 98, 97, 96, ..., 0]

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask] # maks is All false. make no effect.

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,  # type: ignore
                generator=generator,
                **kwargs
                ).prev_sample # type: ignore
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], n_repeat=1, T=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        # print("obs_dict: ", obs_dict.shape)
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        # sample a batch for each sample
        nobs = nobs.unsqueeze(0).repeat(n_repeat, 1, 1, 1).flatten(0, 1)
        
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        if T is None:
            T = self.horizon
        Da = self.action_dim
        

        # print(B, Do, To, T, Da, self.n_action_steps)
        # B = 56, Do = 20, To=2, T=132 (2+128+2), Da=2, self.n_action_steps=128

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            # print("1")
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # print("2") run this branch
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            # print(f'{self.pred_action_steps_only=}') # False
            # print(f'{shape=}') # (56, 132, 2)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # print("3")
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True
            

        # run sampling
        nsamples = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        nsamples = nsamples.view(n_repeat, B//n_repeat, T, -1)
        nsample = nsamples[0]
        
        # unnormalize prediction
        naction_pred_repeat = nsamples[...,:Da]
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # print(f'{nsample.shape=}') (56, 132, 2)
        # print(f'{naction_pred.shape=}') (56, 132, 2)
        # print(f'{action_pred.shape=}') (56, 132, 2)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            # go this way
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
            
        # print(f'{action.shape=}') (56, 64, 2)
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'naction_pred': naction_pred,
            'naction_pred_repeat': naction_pred_repeat,
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end] # type: ignore
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= Prob estimation for dynamic inference  ============
    @torch.no_grad
    def log_prob(self, 
                 obs_dict: Dict[str, torch.Tensor],
                 nactions,
                 a_t_start, at_t_end,
                ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        obs = nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        action = nactions
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        
        # assert nactions.shape[1] == T, "Action horizon should include To"

        # print(B, Do, To, T, Da, self.n_action_steps)
        # B = 56, Do = 20, To=2, T=132 (2+128+2), Da=2 
        
        # zero pad the action seqence to T if shorter than T
        action_horizon = action.shape[1] # (bs, horizon, Da)
        if action_horizon < T:
            padding_length = T - action_horizon
            # F.pad pads in the reverse order of dimensions
            # For a 3D tensor (bs, horizon, Da), padding is applied to the last dimension first
            # So, to pad the second dimension (horizon), we pad (0, 0) for Da and (0, padding_length) for horizon
            # Hence, pad = (0, 0, 0, padding_length)
            action = F.pad(action, (0, 0, 0, padding_length), "constant", 0)
        
        # sample a batch for each sample
        obs = obs.unsqueeze(0).repeat(self.verify_bs, 1, 1, 1).flatten(0, 1)
        action = action.unsqueeze(0).repeat(self.verify_bs, 1, 1, 1).flatten(0, 1)
        
        # build input
        device = self.device
        dtype = self.dtype
 
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
            print("local")
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps

                # print("in trajectory shape: ", start, end, self.n_action_steps, To)
                trajectory = action[:,start:end]
                print("in trajectory shape: ", start, end, self.n_action_steps, To, trajectory.shape, action.shape)

            # print("global")
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else: # GO this branch
            condition_mask = self.mask_generator(trajectory.shape) # ALL FALSE

        # Sample noise that we'll add to the images
        bsz = trajectory.shape[0]
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random timestep for each image
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        # timesteps = (torch.ones( 
        #     (bsz,), device=trajectory.device
        # ) * self.noise_scheduler.config.num_train_timesteps//2).long()
        timesteps = (torch.ones( 
            (bsz,), device=trajectory.device
        ) * self.verify_timestep).long()
        # print(timesteps)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # noisy_trajectory = noise
        
        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        
        # v1: onetstep estimation
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)
        
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")  
        pred = pred[:, a_t_start:at_t_end]
        target = target[:, a_t_start:at_t_end]
       
        log_prob_ =  - F.mse_loss(pred, target, reduction='none').sum(-1) # verify_bs * bs
        log_prob_1 = log_prob_.view(self.verify_bs, B, -1).min(dim=0)[0]
        
        # v2: ddim
        ddim_scheduler = self.ddim_scheduler
        ddim_scheduler.set_timesteps(3)
        _ddim_timesteps = ddim_scheduler.timesteps[ddim_scheduler.timesteps <= self.verify_timestep]
        current_sample = noisy_trajectory.clone()
        for i, t in enumerate(_ddim_timesteps):  # This 't' is now a subset of [50..0]
            
            current_sample[condition_mask] = trajectory[condition_mask]
            model_output = self.model(current_sample, t, 
                local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)
            current_sample = ddim_scheduler.step(model_output, t, current_sample).prev_sample
        
        pred = current_sample[:, a_t_start:at_t_end]
        target = trajectory[:, a_t_start:at_t_end]
       
        log_prob_ =  - F.mse_loss(pred, target, reduction='none').sum(-1) # verify_bs * bs
        log_prob_ddim2 = log_prob_.view(self.verify_bs, B, -1).min(dim=0)[0]
        
        ddim_scheduler = self.ddim_scheduler
        ddim_scheduler.set_timesteps(5)
        _ddim_timesteps = ddim_scheduler.timesteps[ddim_scheduler.timesteps <= self.verify_timestep]
        current_sample = noisy_trajectory.clone()
        for i, t in enumerate(_ddim_timesteps):  # This 't' is now a subset of [50..0]
            
            current_sample[condition_mask] = trajectory[condition_mask]
            model_output = self.model(current_sample, t, 
                local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)
            current_sample = ddim_scheduler.step(model_output, t, current_sample).prev_sample
        
        pred = current_sample[:, a_t_start:at_t_end]
        target = trajectory[:, a_t_start:at_t_end]
       
        log_prob_ =  - F.mse_loss(pred, target, reduction='none').sum(-1) # verify_bs * bs
        log_prob_ddim3 = log_prob_.view(self.verify_bs, B, -1).min(dim=0)[0]
        
        ddim_scheduler = self.ddim_scheduler
        ddim_scheduler.set_timesteps(10)
        _ddim_timesteps = ddim_scheduler.timesteps[ddim_scheduler.timesteps <= self.verify_timestep]
        current_sample = noisy_trajectory.clone()
        for i, t in enumerate(_ddim_timesteps):  # This 't' is now a subset of [50..0]
            
            current_sample[condition_mask] = trajectory[condition_mask]
            model_output = self.model(current_sample, t, 
                local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)
            current_sample = ddim_scheduler.step(model_output, t, current_sample).prev_sample
        
        pred = current_sample[:, a_t_start:at_t_end]
        target = trajectory[:, a_t_start:at_t_end]
       
        log_prob_ =  - F.mse_loss(pred, target, reduction='none').sum(-1) # verify_bs * bs
        log_prob_ddim6 = log_prob_.view(self.verify_bs, B, -1).min(dim=0)[0]
        
        ddim_scheduler = self.ddim_scheduler
        ddim_scheduler.set_timesteps(100)
        _ddim_timesteps = ddim_scheduler.timesteps[ddim_scheduler.timesteps <= self.verify_timestep]
        current_sample = noisy_trajectory.clone()
        for i, t in enumerate(_ddim_timesteps):  # This 't' is now a subset of [50..0]
            
            current_sample[condition_mask] = trajectory[condition_mask]
            model_output = self.model(current_sample, t, 
                local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)
            current_sample = ddim_scheduler.step(model_output, t, current_sample).prev_sample
        
        pred = current_sample[:, a_t_start:at_t_end]
        target = trajectory[:, a_t_start:at_t_end]
       
        log_prob_ =  - F.mse_loss(pred, target, reduction='none').sum(-1) # verify_bs * bs
        log_prob_ddim50 = log_prob_.view(self.verify_bs, B, -1).min(dim=0)[0]

        
        # ddpm
        # scheduler = self.noise_scheduler
        # scheduler.set_timesteps(self.num_inference_steps)
        # current_sample = noisy_trajectory

        # for t in scheduler.timesteps:
        #     # 1. apply conditioning
        #     trajectory[condition_mask] = condition_data[condition_mask] # maks is All false. make no effect.

        #     # 2. predict model output
        #     model_output = model(trajectory, t, 
        #         local_cond=local_cond, global_cond=global_cond) # local cond is none; global cond is flattened (o_t-1, o_t)

        #     # 3. compute previous image: x_t -> x_t-1
        #     trajectory = scheduler.step(
        #         model_output, t, trajectory,  # type: ignore
        #         generator=generator,
        #         **kwargs
        #         ).prev_sample # type: ignore
        
        return {
            "onestep": log_prob_1,
            "ddim2": log_prob_ddim2,
            "ddim3": log_prob_ddim3,
            "ddim6": log_prob_ddim6,
            "ddim50": log_prob_ddim50,
        }
       
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        # print("into it? ")
        assert 'valid_mask' not in batch
        mask = batch['mask']
        batch = {key: batch[key] for key in ['obs', 'action']}
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs'] # type: ignore # [bs, 2+130, 20]
        action = nbatch['action'] # type: ignore # [bs, 2+130, 2]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
            # print(self.n_obs_steps, self.n_action_steps)
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        random_trajectory_length = torch.randint(low=4, high=nbatch['action'].shape[1]//4, size=(1,))[0].item() * 4
        # random_trajectory_length = random.choice([16, 32, 48, 64, 72, 96, 112, 128])
        # print("trajectory and mask shape", trajectory.shape, mask.shape)
        trajectory = trajectory[:,:random_trajectory_length]
        mask = mask[:,:random_trajectory_length]
        # print("trajectory and mask shape", trajectory.shape, mask.shape)


        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else: # GO this branch
            condition_mask = self.mask_generator(trajectory.shape) # ALL FALSE

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # batch_size, seq_len, action_dim = pred.shape
        # mask_lengths = [34, 66, 98, 132]
        # loss_mask = torch.zeros_like(pred, dtype=torch.bool)
        # for b in range(batch_size):
        #     chosen_length = random.choice(mask_lengths)
        #     loss_mask[b, :chosen_length, :] = 1
        #     # print(loss_mask[b], sum(loss_mask[b]))
        # print(mask.shape, torch.sum(mask, dim=1))
        # masked_pred = pred * loss_mask
        # masked_target = target * loss_mask
        masked_pred = pred * mask
        masked_target = target * mask

        # print(pred.shape, target.shape)
        loss = F.mse_loss(masked_pred, masked_target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


    def train_action(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
            print("local")
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps

                # print("in trajectory shape: ", start, end, self.n_action_steps, To)
                trajectory = action[:,start:end]
                print("in trajectory shape: ", start, end, self.n_action_steps, To, trajectory.shape, action.shape)

            print("global")
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        print("before sample noise: ", trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        print("noise_trajectory.shape: ", noisy_trajectory.shape, self.obs_as_global_cond, self.obs_as_local_cond, self.n_obs_steps, self.n_action_steps+self.n_obs_steps)

        return self.normalizer['action'].unnormalize(pred) #[:,self.n_obs_steps:self.n_action_steps+self.n_obs_steps]
