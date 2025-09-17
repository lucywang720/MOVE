if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil
import torch.nn as nn

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

def load_policy(checkpoint):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    policy: DiffusionUnetLowdimPolicy
    policy = hydra.utils.instantiate(cfg.policy)

    return policy, cfg

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        policy4, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.26/Ta4_01.52.36_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0100-test_mean_score=0.893.ckpt')
        policy8, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.28/Ta8_09.09.07_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=1550-test_mean_score=0.897.ckpt')
        policy16, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.25/Ta20_06.44.57_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0950-test_mean_score=0.886.ckpt')
        policy32, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.25/Ta36_06.43.31_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=1450-test_mean_score=0.907.ckpt')
        policy64, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.25/Ta68_06.43.14_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0900-test_mean_score=0.824.ckpt')
        policy128, cfg = load_policy('/diffusion_policy/data/outputs/2024.11.25/03.35.32_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints/epoch=0600-test_mean_score=0.702.ckpt')
        
        policies = [policy4, policy8, policy16, policy32, policy64, policy128]

        self.model = nn.Sequential(*policies)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        for i in range(len(self.model)):
            self.model[i].set_normalizer(normalizer)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        for i in range(len(self.model)):
            self.model[i].to(device)
            self.model[i].eval()

        # save batch for sampling
        train_sampling_batch = None

        action4, action8, action16, action32, action64, action128 = [], [], [], [], [], []

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        # print(type(self.model))
                        # raw_loss = self.model.compute_loss(batch)
                        print(batch.keys())
                        # print(batch['obs'].shape, batch['obs'][0].shape)
                        # print(batch['obs'].shape, batch['obs'][0].shape)
                        # batch['obs'] = batch['obs'][:,-2:]
                        # batch['obs_mask'] = batch['obs_mask'][:,:2]
                        with torch.no_grad():
                            # action4.append(self.model[0].predict_action(batch)['action'])
                            # # print(action4[-1], self.model[0].train_action(batch), self.model[0].train_action(batch).shape)
                            # print(batch['action'][:,1:5])
                            # # print(action4[-1][:,-1]-action4[-1][:,0], torch.norm(action4[-1][:,-1]-action4[-1][:,0],dim=-1))
                            # action8.append(self.model[1].predict_action(batch)['action'])
                            # action16.append(self.model[2].predict_action(batch)['action'])
                            # action32.append(self.model[3].predict_action(batch)['action'])
                            # action64.append(self.model[4].predict_action(batch)['action'])
                            # action128.append(self.model[5].predict_action(batch)['action'])

                            # self.model[1].obs_as_global_cond = True
                            # print("?", self.model[1].train_action(batch).shape, self.model[1].predict_action(batch)['action'].shape)
                            
                            for i in range(6):
                                self.model[i].pred_action_steps_only = True


                            action4.append(self.model[0].train_action(batch))
                            action8.append(self.model[1].train_action(batch))
                            action16.append(self.model[2].train_action(batch))
                            action32.append(self.model[3].train_action(batch))
                            action64.append(self.model[4].train_action(batch))
                            action128.append(self.model[5].train_action(batch))

                            # print(action4[-1].shape, action8[-1].shape, action16[-1].shape, action32[-1].shape)
                            print(action16[-1].shape)
                            # print(torch.norm(action4[-1][:,-1]-action4[-1][:,0],dim=-1))
                            # print('-------------')
                            # print(torch.norm(action8[-1][:,-1]-action8[-1][:,0],dim=-1))
                            # print('-------------')
                            # print(torch.norm(action16[-1][:,-1]-action16[-1][:,0],dim=-1))

                            # print(self.model[5].predict_action(batch)['action'])

                            # print("shape: ", self.model[0].predict_action(batch)['action'].shape)

                        print(batch_idx)
                        if batch_idx % 40 == 0 and batch_idx != 0:
                            action4 = torch.stack(action4)
                            print(action4.shape)
                            action8 = torch.stack(action8)
                            print(action8.shape)
                            action16 = torch.stack(action16)
                            print(action16.shape)
                            action32 = torch.stack(action32)
                            print(action32.shape)
                            action64 = torch.stack(action64)
                            print(action64.shape)
                            action128 = torch.stack(action128)
                            print(action128.shape)

                            torch.save(action4, '/diffusion_policy/action4-tmp.pt')
                            torch.save(action8, '/diffusion_policy/action8-tmp.pt')
                            torch.save(action16, '/diffusion_policy/action16-tmp-2.pt')
                            torch.save(action32, '/diffusion_policy/action32-tmp.pt')
                            torch.save(action64, '/diffusion_policy/action64-tmp.pt')
                            torch.save(action128, '/diffusion_policy/action128-tmp.pt')

                            exit(0)
                
                

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
