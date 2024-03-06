import os
import torch
from logger_set import LOG
from absl import app
import wandb
from utils import set_seed
from path import Path
from pprint import pprint
#from pathlib import Path

from diffusion import GeometryDiffusionScheduler
from accelerate import Accelerator
from data_loaders.canva import CanvaLayout
from trainers.cal_trainer import TrainLoopCAL
from models.CAL import CAL_6, CAL_518
from utils import str2bool

import ml_collections
import argparse

def main(*args, **kwargs):
    config = init_job()
    pprint(vars(config))
    LOG.info("Loading data.")

    if config.dataset == 'canva':
        train_data = CanvaLayout(config.train_json, config.train_clip_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size, z_scaling_size = config.z_scaling_size, mean_0 = config.mean_0)
        val_data = CanvaLayout(config.val_json, config.val_clip_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size, z_scaling_size = config.z_scaling_size, mean_0 = config.mean_0)
    else:
        raise NotImplementedError

    accelerator = Accelerator(
        split_batches=config.optimizer.split_batches, #큰 배치를 더 작은 배치로 나누는 역할 
        gradient_accumulation_steps=config.optimizer.gradient_accumulation_steps, #여러 배치에 걸쳐 그래디언트를 누적한 다음 업데이트를 수행
        mixed_precision=config.optimizer.mixed_precision, #혼합 정밀도 훈련은 계산 효율성을 높이기 위해 부동 소수점 연산에 다른 정밀도(예: float16과 float32)를 혼합 사용
        project_dir=config.log_dir, #logs폴더가 project dir가 되고 workdir가 test라 logs/test가 working directory가 되는것!
    )
    LOG.info(accelerator.state)
    LOG.info("Creating model and diffusion process...")

    if config.image_pred_ox:
        model = CAL_518()
    else:
        model = CAL_6()
        
    model = model.to(accelerator.device)
    noise_scheduler = GeometryDiffusionScheduler(seq_max_length=config.max_num_comp,
                                              device=accelerator.device,
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type=config.diffusion_mode,
                                              clip_sample=False)
    LOG.info("Starting training...")
    TrainLoopCAL(accelerator=accelerator, model=model, diffusion=noise_scheduler,
                 train_data=train_data, val_data=val_data, opt_conf=config.optimizer,
                 log_interval=config.log_interval, save_interval=config.save_interval,
                 device=accelerator.device, resume_from_checkpoint=config.resume_from_checkpoint,
                 diffusion_mode = config.diffusion_mode, scaling_size = config.scaling_size,
                 z_scaling_size=config.z_scaling_size, mean_0 = config.mean_0, loss_weight=config.loss_weight,
                 is_cond = config.is_cond, t_sampling = config.t_sampling, image_pred_ox = config.image_pred_ox).train()


def init_job():
    parser = argparse.ArgumentParser()
    
    # Basic Configuration
    parser.add_argument('--dataset_path', type=str, default="dataset")
    parser.add_argument('--train_json', type=str, default="dataset/train_canva.json")
    parser.add_argument('--train_clip_json', type=str, default="dataset/train_clip.json")
    parser.add_argument('--val_json', type=str, default="dataset/val_canva.json")
    parser.add_argument('--val_clip_json', type=str, default="dataset/val_clip.json")
    parser.add_argument('--resume_from_checkpoint', default=None)
    parser.add_argument('--dataset', type=str, default="canva")
    parser.add_argument('--max_num_comp', type=int, default=40)

    # Training Information
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scaling_size', type=int, default=1)
    parser.add_argument('--z_scaling_size', type=int, default=1)
    parser.add_argument('--mean_0',  type=str2bool, default=True) # Default is False
    parser.add_argument('--is_cond', type=str2bool, default=False) # Default is True
    parser.add_argument('--t_sampling', type=str, default="uniform")

    # Data Specific
    parser.add_argument('--categories_num', type=int, default=7)

    # Model Mode
    parser.add_argument('--rz_ox', type=str2bool, default=True) # Default is True
    parser.add_argument('--image_pred_ox', type=str2bool, default=True) # Default is True

    parser.add_argument('--loss_weight', nargs='+', type=float, default=[1,1,1,1.5])

    # Model Specific
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout_r', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default="gelu")
    parser.add_argument('--cond_emb_size', type=int, default=224)
    parser.add_argument('--cls_emb_size', type=int, default=64)

    # Diffusion Specific
    parser.add_argument('--num_cont_timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default="squaredcos_cap_v2")
    parser.add_argument('--diffusion_mode', type=str, default="epsilon")

    # Logging Information
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=50000)

    #directories
    parser.add_argument('--work_dir', type=str, default="test")
    parser.add_argument('--log_dir', type=str, default="logs/test")
    parser.add_argument('--ckpt_dir', type=str, default="logs/test/checkpoints")
    parser.add_argument('--samples_dir', type=str, default="logs/test/samples")
    
    #optimizer_args
    parser.add_argument('--epoch', type=int, default="2000")
    parser.add_argument('--batch_size', type=int, default="64")
    
    # Parse arguments
    config = parser.parse_args()

    # Convert string paths to Path objects if needed
    config.log_dir = Path(config.log_dir)
    config.dataset_path = Path(config.dataset_path)
    config.train_json = Path(config.train_json)
    config.train_clip_json = Path(config.train_clip_json)
    config.val_json = Path(config.val_json)
    config.val_clip_json = Path(config.val_clip_json)
    
    # Optimizer
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.num_gpus = torch.cuda.device_count()

    config.optimizer.mixed_precision = 'no'
    config.optimizer.gradient_accumulation_steps = 1
    config.optimizer.betas = (0.95, 0.999)
    config.optimizer.epsilon = 1e-8
    config.optimizer.weight_decay = 1e-6

    config.optimizer.lr_scheduler = 'cosine'
    config.optimizer.num_warmup_steps = 5_000
    config.optimizer.lr = 0.0001

    config.optimizer.num_epochs = config.epoch
    config.optimizer.batch_size = config.batch_size
    config.optimizer.split_batches = False
    config.optimizer.num_workers = 4

    config.optimizer.lmb = 5

    if config.optimizer.num_gpus == 0:
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    
    set_seed(config.seed)
    
    config_dict = vars(config)
    wandb.init(project='TEST' if config.work_dir == 'test' else 'DLT', name=config.work_dir,
               mode='disabled' if config.work_dir == 'test' else 'online',
               save_code=True, magic=True, config={k: v for k,v in config_dict.items() if k != 'optimizer'})
    wandb.run.log_code(".")
    wandb.config.update(config.optimizer)
    
    return config        

if __name__ == '__main__':
    main()