# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/run.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
import sys
import errno
import json
import time
from collections import OrderedDict

import torch.nn as nn
import torch.optim as optim

import scipy.io as scio

from common.arguments import parse_args
from data.load_noisy_data import load_Dataset_3dhp

from common.nets.load_net import HPE_model

from common.loss import *
from common.utils import *


###################
args = parse_args()
# print(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id[0]

with open(args.config) as f:
    config = json.load(f)

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)


params = config['params']
if args.dataset is not None:
    params['dataset'] = args.dataset
if args.model is not None:
    params['model'] = args.model
if args.keypoints is not None:
    params['keypoints'] = args.keypoints
if args.subjects_train is not None:
    params['subjects_train'] = args.subjects_train
if args.subjects_test is not None:
    params['subjects_test'] = args.subjects_test
if args.actions is not None:
    params['actions'] = args.actions
if args.stride is not None:
    params['stride'] = int(args.stride)
if args.timesteps is not None:
    params['timesteps'] = int(args.timesteps)
if args.sampling_timesteps is not None:
    params['sampling_timesteps'] = int(args.sampling_timesteps)
if args.epochs is not None:
    params['epochs'] = int(args.epochs)
if args.batch_size is not None:
    params['batch_size'] = int(args.batch_size)
if args.batch_size_eval is not None:
    params['batch_size_eval'] = int(args.batch_size_eval)
if args.learning_rate is not None:
    params['learning_rate'] = float(args.learning_rate)
if args.lr_decay is not None:
    params['lr_decay'] = float(args.lr_decay)
if args.data_augmentation is not None:
    params['data_augmentation'] = True if args.data_augmentation=='True' else False
if args.test_time_augmentation is not None:
    params['test_time_augmentation'] = True if args.test_time_augmentation=='True' else False
if args.number_of_frames is not None:
    params['number_of_frames'] = int(args.number_of_frames)
if args.out_all is not None:
    params['out_all'] = True if args.out_all=='True' else False
if args.transformer_depth is not None:
    params['transformer_depth'] = int(args.transformer_depth)
if args.clip_denoised is not None:
    params['clip_denoised'] = True if args.clip_denoised=='True' else False
if args.with_time_emb is not None:
    params['with_time_emb'] = True if args.with_time_emb=='True' else False
if args.patience is not None:
    params['patience'] = int(args.patience)
if args.ddim_sampling_eta is not None:
    params['ddim_sampling_eta'] = float(args.ddim_sampling_eta)
if args.embed_dim is not None:
    params['embed_dim'] = int(args.embed_dim)
if args.beta_schedule is not None:
    params['beta_schedule'] = str(args.beta_schedule)
if args.max_time is not None:
    params['max_time'] = float(args.max_time)
if args.pretrained_model is not None:
    params['pretrained_model'] = str(args.pretrained_model)
else:
    params['pretrained_model'] = None
if args.use_both_loss is not None:
    params['use_both_loss'] = True if args.use_both_loss=='True' else False
if args.clip_loss is not None:
    params['clip_loss'] = True if args.clip_loss=='True' else False

for params_key in params.keys():
    setattr(args, params_key, params[params_key])

if params['out_all']:
    from common.conditional_diffusion_ddim_normal_directPredict_variableLoss_both_crossFrames import GaussianDiffusion
else:
    from common.conditional_diffusion_s2f_ddim_normal_directPredict_variableLoss_both_crossFrames import \
            GaussianDiffusion

DATASET_NAME = args.dataset
MODEL_NAME = args.model

write_config_file = os.path.join(args.checkpoint, 'config_'+time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))
write_log_file = os.path.join(args.checkpoint, 'log_'+time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))
write_results_file = os.path.join(args.checkpoint, 'results_'+time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y'))

print('Loading {}...'.format(DATASET_NAME))
if args.dataset == '3dhp':
    from common.mpiinf3dhp_dataset import MPIINF3DHPDataset
    dataset = MPIINF3DHPDataset(args)
else:
    raise KeyError('Invalid dataset')

receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
with open(write_log_file + '.txt', 'a') as f:
    f.write(
        """INFO: Receptive field: {} frames\n""" \
            .format(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = np.inf
min_train_loss = np.inf
min_val_loss = np.inf
best_epoch = 0

test_dataset = load_Dataset_3dhp(args, dataset._test, pos_3d_min=dataset._pos_3d_min, pos_3d_max=dataset._pos_3d_max,
                            split='test', noise_std=args.test_extra_noise_std, joint_drop_rate=args.test_joint_drop)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size_eval),
                                              shuffle=False, num_workers=args.workers, drop_last=False,
                                              pin_memory=True)
num_joints = dataset.num_joints
joints_left, joints_right = dataset.joints_left, dataset.joints_right

#########################################HPE_model
in_chans = 2
model_pos = HPE_model(args.model)(num_frame=receptive_field, num_joints=num_joints, in_chans=in_chans, embed_dim=params['embed_dim'],
                                  depth=params['transformer_depth'], num_heads=8, mlp_ratio=2.,
                                  qkv_bias=True, qk_scale=None,drop_path_rate=0.1,with_time_emb=args.with_time_emb)

model_diffusion = GaussianDiffusion(model=model_pos,
                                    timesteps=params['timesteps'],
                                    sampling_timesteps=params['sampling_timesteps'],
                                    loss_type='l2',
                                    clip_denoised=args.clip_denoised,
                                    beta_schedule=params['beta_schedule'],
                                    ddim_sampling_eta=params['ddim_sampling_eta'],
                                    clipLoss=params['clip_loss'])

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

with open(write_config_file + '.txt', 'w') as f:
    f.write(
        """Dataset: {},\nModel: {}\n\nparams={}\n\n\nTotal Parameters: {}\n\n""".format(DATASET_NAME,
                                                                                        MODEL_NAME,
                                                                                        params,
                                                                                        model_params))

if torch.cuda.is_available():
    model_diffusion = nn.DataParallel(model_diffusion)
    model_diffusion = model_diffusion.cuda()

if args.resume or args.evaluate or params['pretrained_model'] is not None:
    if args.resume or args.evaluate:
        chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    else:
        chk_filename = params['pretrained_model']
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    new_model_diffusion_checkpoint = OrderedDict()
    for k, v in checkpoint['model_diffusion'].items():
        if 'alphas' in k:
            print('skip {}'.format(k))
            continue
        else:
            name = k
            new_model_diffusion_checkpoint[name] = v
    model_diffusion.load_state_dict(new_model_diffusion_checkpoint, strict=False)

with open(write_log_file + '.txt', 'a') as f:
    f.write(
        """INFO: Testing on {} frames\n""" \
            .format(test_dataset.generator.num_frames()))


###################

if not args.evaluate:

    lr = params['learning_rate']
    optimizer = optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = params['lr_decay']
    losses_3d_train = []
    losses_3d_valid = []
    losses_pose_train = []
    losses_pose_valid = []

    num_epochs_patience = params['patience']
    curr_step = 0
    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    train_dataset = load_Dataset_3dhp(args, dataset._train, pos_3d_min=dataset._pos_3d_min, pos_3d_max=dataset._pos_3d_max,
                                split='train', noise_std=args.train_extra_noise_std, joint_drop_rate=args.train_joint_drop)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers,
                                                   drop_last=True, pin_memory=True)

    train_val_dataset = load_Dataset_3dhp(args, dataset._train, pos_3d_min=dataset._pos_3d_min,
                                          pos_3d_max=dataset._pos_3d_max,
                                          split='val', noise_std=args.train_extra_noise_std,
                                          joint_drop_rate=args.train_joint_drop)

    with open(write_log_file + '.txt', 'a') as f:
        f.write(
            """INFO: Training on {} frames\n""" \
                .format(train_val_dataset.generator.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch'] + 1
        if checkpoint['best_epoch'] is not None:
            best_epoch = checkpoint['best_epoch']
            min_loss = checkpoint['min_loss']
            min_train_loss = checkpoint['min_train_loss']

        lr = checkpoint['lr'] * lr_decay
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
            train_dataloader.dataset.generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            with open(write_log_file + '.txt', 'a') as f:
                f.write(
                    """WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.\n""")


    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    start_training_time = time.time()
    # Pos model only
    while epoch < args.epochs:
        start_time = time.time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        epoch_loss_pose_train = 0
        N = 0
        N_semi = 0
        model_diffusion.train()

        for batch_id, (_, inputs_3d, inputs_3d_norm, inputs_2d, _, target_mask, _, _, _, _, _) in enumerate(train_dataloader):
            target_mask = target_mask.view(-1)

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_3d_norm = inputs_3d_norm.cuda()
                inputs_2d = inputs_2d.cuda()

            optimizer.zero_grad()

            # Predict 3D poses
            loss_pose, predicted_3d_pos = model_diffusion(clean_3d_pose=inputs_3d_norm,
                                                          noisy_2d_pose=inputs_2d)
            loss_pose = loss_pose.mean()
            inputs_3d = inputs_3d.view(-1, num_joints, 3)
            inputs_3d = inputs_3d[target_mask == True, :, :].unsqueeze(1)

            epoch_loss_pose_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_pose.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_pose.backward()

            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N)
        losses_pose_train.append(epoch_loss_pose_train / N)

        # End-of-epoch evaluation
        with torch.no_grad():
            model_diffusion.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            epoch_loss_pose_valid = 0
            N = 0
            if not args.no_eval:
                for batch_id, (_, inputs_3d, inputs_3d_norm, inputs_2d, inputs_2d_flip, target_mask, _, _, _, _, _) in enumerate(test_dataloader):
                    target_mask = target_mask.view(-1)
                    inputs_3d_norm_flip = inputs_3d_norm.clone()
                    inputs_3d_norm_flip[:, :, :, 0] *= -1
                    inputs_3d_norm_flip[:, :, joints_left + joints_right] = inputs_3d_norm_flip[:, :,
                                                                            joints_right + joints_left]
                    if torch.cuda.is_available():
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                        inputs_3d = inputs_3d.cuda()
                        inputs_3d_norm_flip = inputs_3d_norm_flip.cuda()

                    loss_pose, predicted_3d_pos = model_diffusion(clean_3d_pose=inputs_3d_norm,
                                                                  noisy_2d_pose=inputs_2d)
                    loss_pose = loss_pose.mean()
                    loss_pose_flip, predicted_3d_pos_flip = model_diffusion(clean_3d_pose=inputs_3d_norm_flip,
                                                                            noisy_2d_pose=inputs_2d_flip)
                    loss_pose_flip = loss_pose_flip.mean()
                    loss_pose = (loss_pose + loss_pose_flip) / 2.0
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]

                    predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2.0
                    predicted_3d_pos = test_dataloader.dataset.reverse_norm_3d_pose(predicted_3d_pos)
                    predicted_3d_pos = predicted_3d_pos.view(-1, num_joints, 3)
                    predicted_3d_pos = predicted_3d_pos[target_mask == True, :, :].unsqueeze(1)
                    inputs_3d = inputs_3d.view(-1, num_joints, 3)
                    inputs_3d = inputs_3d[target_mask == True, :, :].unsqueeze(1)

                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    epoch_loss_pose_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_pose.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                losses_pose_valid.append(epoch_loss_pose_valid / N)

        elapsed = (time.time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f loss_pose_train %f' % (
                epoch,
                elapsed,
                lr,
                losses_3d_train[-1],
                losses_pose_train[-1]))
            with open(write_log_file + '.txt', 'a') as f:
                f.write(
                    """Epoch: {}, Time(mins): {}, learning_rate: {:.8f}, Train/_3d_loss: {:.8f}, Train/_pose_loss: {:.8f}\n""" \
                        .format(epoch,
                                elapsed,
                                lr,
                                losses_3d_train[-1],
                                losses_pose_train[-1]))
        else:
            print(
                '[%d] time %.2f lr %f 3d_train %f 3d_valid %f loss_pose_train %f loss_pose_valid %f' % (
                    epoch,
                    elapsed,
                    lr,
                    losses_3d_train[-1],
                    losses_3d_valid[-1],
                    losses_pose_train[-1],
                    losses_pose_valid[-1]))
            with open(write_log_file + '.txt', 'a') as f:
                f.write(
                    """Epoch: {}, Time(mins): {}, learning_rate: {:.8f}, Train/_loss: {:.8f}, Test/_loss: {:.8f}, Train/_pose_loss: {:.8f}, Test/_pose_loss: {:.8f}\n""" \
                        .format(epoch,
                                elapsed,
                                lr,
                                losses_3d_train[-1],
                                losses_3d_valid[-1],
                                losses_pose_train[-1],
                                losses_pose_valid[-1]))

        if (epoch + 1) % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'best_epoch': best_epoch,
                'min_loss': min_loss,
                'min_train_loss': min_train_loss,
                'lr': lr,
                'random_state': train_dataloader.dataset.generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_diffusion': model_diffusion.state_dict(),
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
        if losses_3d_valid[-1] < min_loss:
            min_loss = losses_3d_valid[-1]
            min_train_loss = losses_3d_train[-1]
            best_epoch = epoch
            print("save best checkpoint")
            torch.save({
                'epoch': best_epoch,
                'best_epoch': best_epoch,
                'min_loss': min_loss,
                'min_train_loss': min_train_loss,
                'lr': lr,
                'random_state': train_dataloader.dataset.generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_diffusion': model_diffusion.state_dict(),
            }, best_chk_path)
            curr_step = 0
        else:
            curr_step += 1

        with open(write_log_file + '.txt', 'a') as f:
            f.write(
                """Best epoch: {}, Best_train/_loss: {:.4f}, Best_val/_loss: {:.8f}, Best_test/_loss: {:.8f}\n""" \
                    .format(best_epoch,
                            min_train_loss,
                            min_val_loss,
                            min_loss))

        if curr_step >= num_epochs_patience:
            print("Early stop!")
            break

        # Stop training after params['max_time'] hours
        if time.time() - start_training_time > params['max_time'] * 3600:
            print('-' * 89)
            print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
            break

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

    with open(write_log_file + '.txt', 'a') as f:
        f.write(
            """Training finished!\n Total time: {}\n""" \
                .format((time.time() - start_training_time) / 60))

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')


data_inference = {}
def evaluate(test_dataloader, seq_name):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    # Evaluate the best model after training
    if not args.evaluate:
        chk_filename = best_chk_path
        print('Loading the best model', chk_filename)
        with open(write_log_file + '.txt', 'a') as f:
            f.write("""Loading the best model at {}\n""".format(chk_filename))
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model_diffusion.load_state_dict(checkpoint['model_diffusion'], strict=False)
        with open(write_log_file + '.txt', 'a') as f:
            f.write("""Loading succeed!\n""")

    with torch.no_grad():

        model_diffusion.eval()

        N = 0
        for batch_id, (
        _, inputs_3d, inputs_3d_norm, inputs_2d, inputs_2d_flip, target_mask, _, _, _, _, _) in enumerate(
            test_dataloader):
            target_mask = target_mask.view(-1)
            inputs_3d_norm_flip = inputs_3d_norm.clone()
            inputs_3d_norm_flip[:, :, :, 0] *= -1
            inputs_3d_norm_flip[:, :, joints_left + joints_right] = inputs_3d_norm_flip[:, :,
                                                                    joints_right + joints_left]

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_3d_norm = inputs_3d_norm.cuda()
                inputs_3d_norm_flip = inputs_3d_norm_flip.cuda()

            # Predict 3D poses
            _, predicted_3d_pos = model_diffusion(clean_3d_pose=inputs_3d_norm,
                                                  noisy_2d_pose=inputs_2d)
            _, predicted_3d_pos_flip = model_diffusion(clean_3d_pose=inputs_3d_norm_flip,
                                                       noisy_2d_pose=inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]

            predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2.0
            predicted_3d_pos = test_dataloader.dataset.reverse_norm_3d_pose(predicted_3d_pos)
            predicted_3d_pos = predicted_3d_pos.view(-1, num_joints, 3)
            predicted_3d_pos = predicted_3d_pos[target_mask == True, :, :].unsqueeze(1)
            inputs_3d = inputs_3d.view(-1, num_joints, 3)
            inputs_3d = inputs_3d[target_mask == True, :, :].unsqueeze(1)


            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().view(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            # Store predictions
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (data_inference[seq_name], predicted_3d_pos.clone().permute(2, 1, 0).numpy()), axis=2)
            else:
                data_inference[seq_name] = predicted_3d_pos.clone().permute(2, 1, 0).numpy()

            predicted_3d_pos = predicted_3d_pos.numpy()

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    print('----' + seq_name + '----')
    with open(write_results_file + '.txt', 'a') as f:
        f.write(
            """'----'{}'----'\n""".format(seq_name))

    e1 = (epoch_loss_3d_pos / N)
    e2 = (epoch_loss_3d_pos_procrustes / N)
    e3 = (epoch_loss_3d_pos_scale / N)
    ev = (epoch_loss_3d_vel / N)
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    with open(write_results_file + '.txt', 'a') as f:
        f.write(
            """Protocol #1 Error (MPJPE): {}mm\nProtocol #2 Error (P-MPJPE): {}mm\nProtocol #3 Error (N-MPJPE): {}mm\nVelocity Error (MPJVE): {}mm\n----------\n""" \
                .format(e1,
                        e2,
                        e3,
                        ev))

    return e1, e2, e3, ev


with open(write_results_file + '.txt', 'w') as f:
    f.write(
        """Dataset: {},\nModel: {}\n\nargs={}\n\n\nTotal Parameters: {}\n\n""".format(DATASET_NAME,
                                                                                      MODEL_NAME,
                                                                                      args,
                                                                                      model_params))
print('Evaluating...')
with open(write_log_file + '.txt', 'a') as f:
    f.write("""Evaluating...\n""")


def run_evaluation():
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []

    for seq_name in args.subjects_test.split(','):

        test_dataset = load_Dataset_3dhp(args, dataset._test, pos_3d_min=dataset._pos_3d_min,
                                         pos_3d_max=dataset._pos_3d_max,
                                         split='test', noise_std=args.test_extra_noise_std,
                                         joint_drop_rate=args.test_joint_drop,
                                         seq_filter=seq_name)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(args.batch_size_eval),
                                                      shuffle=False, num_workers=args.workers, drop_last=False,
                                                      pin_memory=True)

        e1, e2, e3, ev = evaluate(test_dataloader, seq_name)
        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)

    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
    print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
    print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

    with open(write_results_file + '.txt', 'a') as f:
        f.write(
            """Protocol #1   (MPJPE) action-wise average: {}mm\nProtocol #2 (P-MPJPE) action-wise average: {}mm\nProtocol #3 (N-MPJPE) action-wise average: {}mm\nVelocity      (MPJVE) action-wise average: {}mm\n""" \
                .format(round(np.mean(errors_p1), 1),
                        round(np.mean(errors_p2), 1),
                        round(np.mean(errors_p3), 1),
                        round(np.mean(errors_vel), 2)))

if not args.by_subject:
    run_evaluation()
    mat_path = os.path.join(args.checkpoint, 'inference_data.mat')
    scio.savemat(mat_path, data_inference)