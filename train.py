# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', message='.*kernel_size exceeds volume extent.*')
from PIL import Image
import itertools
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset2 import CodeDataset, mel_spectrogram, get_dataset_filelist, ECGDataset
from models2 import CodeGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import plot_spectrogram, plot_spectrogram2, scan_checkpoint, load_checkpoint, \
    save_checkpoint, build_env, AttrDict
import ecg_plot 
import torchvision.transforms as transforms 
import wandb
torch.backends.cudnn.benchmark = True
run = wandb.init(
    project = 'ECG_Hifi-GAN',
    job_type = 'train_model',
)

def train(rank, local_rank, a, h):
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(local_rank))

    generator = CodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank],
            find_unused_parameters=('f0_quantizer' in h),
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[local_rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[local_rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate,
                                betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(h)

    trainset = CodeDataset(training_filelist, h.segment_size, h.code_hop_size, h.n_fft, h.num_mels, h.hop_size,
                           h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
                           device=device, f0=h.get('f0', None), multispkr=h.get('multispkr', None),
                           f0_stats=h.get('f0_stats', None),
                           f0_normalize=h.get('f0_normalize', False), f0_feats=h.get('f0_feats', False),
                           f0_median=h.get('f0_median', False), f0_interp=h.get('f0_interp', False),
                           vqvae=h.get('code_vq_params', False))
    
    dataset = ECGDataset(h.n_fft, 40, 500, h.hop_size, h.win_size, h.fmin, h.fmax, device=device)
    
    train_size = int(0.9 * len(dataset))
    validation_size = int(len(dataset) - (train_size))
    
    training_data, validation_data= torch.utils.data.random_split(dataset, [train_size, validation_size])
    
    

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    
    train_loader2 = DataLoader(training_data, num_workers=0, shuffle=True, sampler=None,
                              batch_size=h.batch_size, pin_memory=True, drop_last=True)

    if rank == 0:
        validation_loader2 = DataLoader(validation_data, num_workers=0, shuffle=False, sampler=None,
                                       batch_size=h.batch_size, pin_memory=True, drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader2):
            if rank == 0:
                start_b = time.time()
            # x, y, _, y_mel = batch
            x,y,y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            # y = y.unsqueeze(1) #[4,1,8960]
            # x = {k: torch.autograd.Variable(v.to(device, non_blocking=False)) for k, v in x.items()}
            x = torch.autograd.Variable(x.to(device, non_blocking=False))#[4,56]

            y_g_hat = generator(x.to(device)) 
            if h.get('f0_vq_params', None) or h.get('code_vq_params', None):
                y_g_hat, commit_losses, metrics = y_g_hat

            assert y_g_hat.shape == y.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"
            if h.get('f0_vq_params', None):
                f0_commit_loss = commit_losses[1][0]
                f0_metrics = metrics[1][0]
            if h.get('code_vq_params', None):
                code_commit_loss = commit_losses[0][0]
                code_metrics = metrics[0][0]

            y_g_mel = []
            for i in range(y_g_hat.shape[0]):
                y_g_hat_mel = mel_spectrogram(y_g_hat[i], h.n_fft, 40, 500, h.hop_size,
                                            h.win_size, h.fmin, h.fmax_for_loss)
                y_g_mel.append(y_g_hat_mel)
            
            y_g_hat_mel = torch.stack(y_g_mel)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            if h.get('f0_vq_params', None):
                loss_gen_all += f0_commit_loss * h.get('lambda_commit', None)
            if h.get('code_vq_params', None):
                loss_gen_all += code_commit_loss * h.get('lambda_commit_code', None)

            loss_gen_all.backward()
            optim_g.step()

            with torch.no_grad():
                mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

            print(
                'Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.format(steps,
                                                                                                            loss_gen_all,
                                                                                                            mel_error,
                                                                                                            time.time() - start_b))

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                                    'msd': (msd.module if h.num_gpus > 1 else msd).state_dict(),
                                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                    'steps': steps, 'epoch': epoch})

            # Tensorboard summary logging
            wandb.log({"training/gen_loss_total": loss_gen_all,
                        "training/mel_spec_error": mel_error})
            
            steps += 1
            if steps >= a.training_steps:
                break


        if rank == 0: 
            # Validation
            generator.eval()
            torch.cuda.empty_cache()
            val_err_tot = 0
            with torch.no_grad():
                print(len(validation_loader2))
                for j, batch in enumerate(validation_loader2):
                    x,y,y_mel = batch
                    y = torch.autograd.Variable(y.to(device, non_blocking=False))
                    x = torch.autograd.Variable(x.to(device, non_blocking=False))#[4,56]

                    y_g_hat = generator(x)

                    y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
                    y_g_mel = []
                    for i in range(y_g_hat.shape[0]):
                        y_g_hat_mel = mel_spectrogram(y_g_hat[i], h.n_fft, 40, 500, h.hop_size,
                                                    h.win_size, h.fmin, h.fmax_for_loss)
                        y_g_mel.append(y_g_hat_mel)
                    
                    y_g_hat_mel = torch.stack(y_g_mel)
                    val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                    

                    ecg_plot.plot(y[0].detach().cpu().numpy(), sample_rate=500, title = "Original ECG")
                    ecg_plot.save_as_png("Original_ECG2")
                    wandb.log({"Original ECG": wandb.Image("Original_ECG2.png")})

        
                    x = plot_spectrogram(y_mel[0].cpu())
                    wandb.log({"y_spec": wandb.Image("spectrogram2.png")})
                        
                    ecg_plot.plot(y_g_hat[0].detach().cpu().numpy(), sample_rate=500, title = "Reconstructed ECG")
                    ecg_plot.save_as_png("Reconstructed_ECG2")    
                    wandb.log({"Reconstructed ECG": wandb.Image("Reconstructed_ECG2.png")})

                    x = plot_spectrogram2(y_g_hat_mel[0].cpu())
                    wandb.log({"y_hat_spec": wandb.Image("reconstructed_spectrogram2.png")})


                val_err = val_err_tot / (j + 1)
                wandb.log({"validation/val_error": val_err})

            generator.train()
        generator.train()


        scheduler_g.step()
        scheduler_d.step()

    if rank == 0:
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))




def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan2')
    parser.add_argument('--config', default='configs/LJSpeech/vqvae256_lut.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--training_steps', default=400000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed-world-size', type=int)
    parser.add_argument('--distributed-port', type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and 'WORLD_SIZE' in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ['WORLD_SIZE'])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print('Batch size per GPU :', h.batch_size)
    else:
        rank = 0
        local_rank = 0

    train(rank, local_rank, a, h)


if __name__ == '__main__':
    main()
