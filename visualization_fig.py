# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/common/visualization.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import os
import errno
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch.nn as nn

from common.camera import *
from common.nets.load_net import HPE_model
from common.conditional_diffusion_ddim_normal_directPredict_variableLoss_both_crossFrames import GaussianDiffusion
from common.utils import *

from data.load_noisy_data_viz import load_Dataset


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_2d_figs(keypoints, keypoints_metadata, skeleton, output, size=7):
    """

    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    num_subfig = keypoints.shape[0]
    fig = plt.figure(figsize=(size * num_subfig, size))

    parents = skeleton.parents()
    joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
    colors_2d = np.full(keypoints.shape[1], 'black')
    colors_2d[joints_right_2d] = 'red'

    for i in range(num_subfig):
        subfig_idx = i + 1
        ax_in = fig.add_subplot(1, num_subfig, subfig_idx)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()

        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue

            col = 'red' if j in skeleton.joints_right() else 'black'
            ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                       [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color=col, linewidth=4)

        ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='black', zorder=10, linewidths=4, marker='.')



    fig.tight_layout()
    fig.savefig(output)

    plt.close()


def render_3d_figs(poses, skeleton, azim, output, size=6):
    """
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    num_subfig = poses.shape[0]
    fig = plt.figure(figsize=(size * num_subfig, size))

    radius = 1.7
    parents = skeleton.parents()
    for i in range(num_subfig):
        ax = fig.add_subplot(1, len(poses), i + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        trajectories= poses[i, 0, [0, 1]]
        ax.set_xlim3d([-radius / 2 + trajectories[0], radius / 2 + trajectories[0]])
        ax.set_ylim3d([-radius / 2 + trajectories[1], radius / 2 + trajectories[1]])

        for j, j_parent in enumerate(parents):
            if j_parent == -1:
                continue

            col = 'red' if j in skeleton.joints_right() else 'black'
            ax.plot([poses[i, j, 0], poses[i, j_parent, 0]],
                    [poses[i, j, 1], poses[i, j_parent, 1]],
                    [poses[i, j, 2], poses[i, j_parent, 2]], zdir='z', c=col, linewidth=4)


    fig.tight_layout(w_pad=0.5)
    fig.savefig(output)

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")

    # General arguments
    parser.add_argument('-d', '--dataset', type=str, metavar='NAME', help='target dataset')  # h36m or humaneva
    parser.add_argument('--model', type=str, metavar='NAME',
                        help='Model name')
    parser.add_argument('-k', '--keypoints', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('--extra_noise_std', type=float, default=0.0,
                        help='std of extra Gaussian noise added to 2D pose in viz set')
    parser.add_argument('--joint_drop', type=float, default=0.0,
                        help='drop rate of joints in viz set')
    parser.add_argument('-sviz', '--subjects-visualization', type=str, metavar='LIST',
                        help='visualization subjects separated by comma')
    parser.add_argument('-a', '--actions', type=str, metavar='LIST',
                        help='actions to visualize on, separated by comma, or * for all')
    parser.add_argument('-cam', '--cameras', type=str, metavar='LIST',
                        help='camera id to visualize on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu_id', nargs='+', help="gpu id separated by comma")
    parser.add_argument('--out_all', help='Set True to use all frames as the target')

    # Model arguments
    parser.add_argument('-s', '--stride', type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-b', '--batch-size', type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--timesteps', type=int, help='timesteps of diffusion')
    parser.add_argument('--sampling_timesteps', type=int, help='sampling timesteps of DDIM')
    parser.add_argument('--data-augmentation',
                        help='Set True to use train-time flipping')
    parser.add_argument('--test-time-augmentation',
                        help='Set True to use test-time flipping')
    parser.add_argument('-frame', '--number-of-frames', type=int, metavar='N',
                        help='how many frames used as input')
    parser.add_argument('--transformer_depth', type=int, metavar='N', help='depth of each transformer module')
    parser.add_argument('--clip_denoised', type=bool, help="set True to clip x_start")
    parser.add_argument('--with_time_emb', type=bool, help="set True to use time emb")
    parser.add_argument('--beta_schedule', default='cosine', type=str,
                        help='beta schedule for diffusion model')
    parser.add_argument('--embed_dim', type=int, metavar='N',
                        help='Number of embedding dim')
    parser.add_argument('--ddim_sampling_eta', default=0.0, type=float, metavar='FACTOR',
                        help='ddim sampling eta')

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor (semi-supervised)')

    # Visualization
    parser.add_argument('--viz-video', type=str, default=None, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    # parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=30000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    parser.add_argument('--viz-fps', type=int, metavar='N', help='FPS')
    args = parser.parse_args()

    return args


args = parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id[0]

with open(args.config) as f:
    config = json.load(f)

chk_filename = os.path.join(args.checkpoint, args.evaluate)
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

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
if args.subjects_visualization is not None:
    params['subjects_visualization'] = args.subjects_visualization
if args.actions is not None:
    params['actions'] = args.actions
if args.cameras is not None:
    params['cameras'] = args.cameras
if args.stride is not None:
    params['stride'] = int(args.stride)
if args.timesteps is not None:
    params['timesteps'] = int(args.timesteps)
if args.sampling_timesteps is not None:
    params['sampling_timesteps'] = int(args.sampling_timesteps)
if args.batch_size is not None:
    params['batch_size'] = int(args.batch_size)
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
if args.ddim_sampling_eta is not None:
    params['ddim_sampling_eta'] = float(args.ddim_sampling_eta)
if args.embed_dim is not None:
    params['embed_dim'] = int(args.embed_dim)
if args.beta_schedule is not None:
    params['beta_schedule'] = str(args.beta_schedule)

for params_key in params.keys():
    setattr(args, params_key, params[params_key])

if args.viz_output is None:
    viz_path = os.path.join(args.checkpoint,
                            str(args.number_of_frames)+'f',
                            args.subjects_visualization,
                            args.actions,
                            args.cameras)
    args.viz_output = 'viz.svg'
else:
    viz_output_list = args.viz_output.split('/')
    if len(viz_output_list) == 1:
        viz_path = os.path.join(str(args.number_of_frames) + 'f',
                                args.subjects_visualization,
                                args.actions,
                                args.cameras)
    elif len(viz_output_list) > 1:
        viz_path = os.path.join(*viz_output_list[:-1])
    else:
        raise print('Invalid viz_output {}'.format(args.viz_output))

    args.viz_output = viz_output_list[-1]

try:
    # Create viz results directory if it does not exist
    os.makedirs(viz_path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create viz results directory:', viz_path)


DATASET_NAME = args.dataset
MODEL_NAME = args.model

print('Loading dataset...')
data_root_path = 'data'
dataset_path = data_root_path + '/data_3d_' + args.dataset + '.npz'
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset(data_root_path + '/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
print('Batch size: {}'.format(args.batch_size))

pad = (receptive_field -1) // 2 # Padding on each side

sbj_list = args.subjects_visualization.split(',')
action_filter = None if args.actions == '*' else args.actions.split(',')
camera_filter = None if args.cameras == '*' else list(map(int, args.cameras.split(',')))
viz_fps = dataset.fps() if args.viz_fps is None else args.viz_fps

batch_stop = 0
for curr_sbj in sbj_list:
    for curr_action in action_filter:
        for curr_camera in camera_filter:



            viz_dataset = load_Dataset(args, dataset, data_root_path, [curr_sbj], [curr_action], [curr_camera],
                                       noise_std=args.extra_noise_std, joint_drop_rate=args.joint_drop)
            keypoints_metadata = viz_dataset.keypoints_metadata
            cam = dataset.cameras()[curr_sbj][curr_camera]
            viz_dataloader = torch.utils.data.DataLoader(viz_dataset, batch_size=int(args.batch_size),
                                                         shuffle=False, num_workers=args.workers, drop_last=False,
                                                         pin_memory=True)

            num_joints = viz_dataset.num_joints
            joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

            #########################################HPE_model
            in_chans = 2
            model_pos = HPE_model(args.model)(num_frame=receptive_field, num_joints=num_joints, in_chans=in_chans,
                                              embed_dim=params['embed_dim'],
                                              depth=params['transformer_depth'], num_heads=8,
                                              mlp_ratio=2.,
                                              qkv_bias=True, qk_scale=None, drop_path_rate=0.,
                                              with_time_emb=True)

            model_diffusion = GaussianDiffusion(model=model_pos,
                                                timesteps=params['timesteps'],
                                                sampling_timesteps=params['sampling_timesteps'],
                                                loss_type='l2',
                                                clip_denoised=args.clip_denoised,
                                                beta_schedule=params['beta_schedule'],
                                                ddim_sampling_eta=params['ddim_sampling_eta'],
                                                clipLoss=True)


            model_params = 0
            for parameter in model_pos.parameters():
                model_params += parameter.numel()
            print('INFO: Trainable parameter count:', model_params)

            if torch.cuda.is_available():
                model_diffusion = nn.DataParallel(model_diffusion)
                model_diffusion = model_diffusion.cuda()

            model_diffusion.load_state_dict(checkpoint['model_diffusion'], strict=False)

            anim_output = {}
            anim_output['Ground truth'] = []
            anim_output['Reconstruction'] = []
            error_output = {}
            error_output['Reconstruction'] = []

            error_reverse_diffusion = {}
            error_reverse_diffusion['Reverse diffusion process'] = []
            error_reverse_diffusion['Start 3D pose est.'] = []

            anim_diffusion_output = {}
            anim_reverse_diffusion_input = {}
            anim_reverse_diffusion_output = {}
            anim_reverse_diffusion_output[
                'Reverse diffusion process start'] = []
            anim_reverse_diffusion_output[
                'Reverse diffusion process middle2'] = []
            anim_reverse_diffusion_output[
                'Reverse diffusion process middle'] = []
            anim_reverse_diffusion_output[
                'Reverse diffusion process end'] = []
            anim_reverse_diffusion_input[
                'Reverse diffusion process input 2D'] = []

            input_keypoints = []
            print('Rendering...')
            with torch.no_grad():
                model_diffusion.eval()
                for batch_id, (
                _, trajectory, inputs_3d, inputs_3d_norm, inputs_2d, inputs_2d_flip, target_mask, inputs_3d_frame_id,
                inputs_2D_frame_id, _, action, subject, _, _, cam_ind) in enumerate(viz_dataloader):
                    # cam_ind is a tensor!

                    print('Current batch id {}'.format(batch_id))
                    num_seq = inputs_3d.size(0)
                    len_seq = inputs_3d.size(1)
                    print('num_seq {}'.format(num_seq))
                    print('len_seq {}'.format(len_seq))

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

                    if batch_id == 0:
                        diffusion_3d_pos, t_list = model_diffusion.module.get_noisy_pose(inputs_3d_norm.clone(), num_sample=40)
                        diffusion_3d_pos = diffusion_3d_pos.cpu()
                        num_sample = diffusion_3d_pos.size(4)
                        diffusion_3d_pos += trajectory.unsqueeze(-1).repeat(1, 1, 1, 1, num_sample)
                        diffusion_3d_pos = diffusion_3d_pos.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0]
                        diffusion_3d_pos = viz_dataloader.dataset.reverse_norm_3d_pose(diffusion_3d_pos)
                        anim_diffusion_output['Diffusion process'] = diffusion_3d_pos

                    inputs_3d_norm_flip = inputs_3d_norm.clone()
                    inputs_3d_norm_flip[:, :, :, 0] *= -1
                    inputs_3d_norm_flip[:, :, joints_left + joints_right] = inputs_3d_norm_flip[:, :,
                                                                            joints_right + joints_left]

                    # Predict 3D poses
                    _, predicted_3d_pos, reverse_diffusion_3d_pos, start_3d_pos_est = model_diffusion(
                        clean_3d_pose=inputs_3d_norm,
                        noisy_2d_pose=inputs_2d,
                        output_reverse_diffusion_3d=True)
                    _, predicted_3d_pos_flip = model_diffusion(clean_3d_pose=inputs_3d_norm_flip,
                                                               noisy_2d_pose=inputs_2d_flip,
                                                               output_reverse_diffusion_3d=False)
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]
                    predicted_3d_pos = (predicted_3d_pos + predicted_3d_pos_flip) / 2.0
                    predicted_3d_pos = viz_dataloader.dataset.reverse_norm_3d_pose(predicted_3d_pos)

                    inputs_3d_one_frame = inputs_3d[0, 0, :, :]

                    reverse_diffusion_3d_pos = viz_dataloader.dataset.reverse_norm_3d_pose(reverse_diffusion_3d_pos)
                    num_sample = reverse_diffusion_3d_pos.size(4)

                    inputs_3d_one_frame = inputs_3d_one_frame.unsqueeze(0).repeat(num_sample, 1, 1)
                    print('inputs_3d_one_frame {}'.format(inputs_3d_one_frame.size()))
                    print('reverse_diffusion_3d_pos {}'.format(reverse_diffusion_3d_pos.size()))


                    reverse_diffusion_3d_pos = reverse_diffusion_3d_pos.cpu()
                    reverse_diffusion_3d_pos += trajectory.unsqueeze(-1).repeat(1, 1, 1, 1, num_sample)
                    reverse_diffusion_3d_pos = reverse_diffusion_3d_pos.permute(4, 1, 2, 3, 0)
                    reverse_diffusion_3d_pos_start = reverse_diffusion_3d_pos[0, 0, :, :, 0]
                    reverse_diffusion_3d_pos_middle2 = reverse_diffusion_3d_pos[-3, 0, :, :, 0]
                    reverse_diffusion_3d_pos_middle = reverse_diffusion_3d_pos[-2, 0, :, :, 0]
                    reverse_diffusion_3d_pos_end = reverse_diffusion_3d_pos[-1, 0, :, :, 0]

                    anim_reverse_diffusion_output[
                        'Reverse diffusion process start'].append(reverse_diffusion_3d_pos_start)
                    anim_reverse_diffusion_output[
                        'Reverse diffusion process middle2'].append(reverse_diffusion_3d_pos_middle2)
                    anim_reverse_diffusion_output[
                        'Reverse diffusion process middle'].append(reverse_diffusion_3d_pos_middle)
                    anim_reverse_diffusion_output[
                        'Reverse diffusion process end'].append(reverse_diffusion_3d_pos_end)
                    anim_reverse_diffusion_input[
                        'Reverse diffusion process input 2D'].append(inputs_2d[0, 0, :, :].clone().cpu())

                    start_3d_pos_est = viz_dataloader.dataset.reverse_norm_3d_pose(start_3d_pos_est)
                    num_sample = start_3d_pos_est.size(4)


                    start_3d_pos_est = start_3d_pos_est.cpu()
                    start_3d_pos_est += trajectory.unsqueeze(-1).repeat(1, 1, 1, 1, num_sample)
                    start_3d_pos_est = start_3d_pos_est.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0]
                    anim_reverse_diffusion_output['Start 3D pose est.'] = start_3d_pos_est

                    predicted_3d_pos = predicted_3d_pos.cpu()
                    predicted_3d_pos += trajectory
                    predicted_3d_pos = predicted_3d_pos.view(-1, num_joints, 3)[target_mask == True, :, :]
                    anim_output['Reconstruction'].append(predicted_3d_pos)



                    inputs_3d = inputs_3d.cpu()
                    inputs_3d += trajectory
                    ground_truth = inputs_3d.view(-1, num_joints, 3)[target_mask == True, :, :]
                    anim_output['Ground truth'].append(ground_truth)
                    inputs_2d = inputs_2d.cpu()
                    inputs_2d = inputs_2d.view(-1, num_joints, 2)[target_mask == True, :, :]
                    input_keypoints.append(inputs_2d)

                    if batch_id >= batch_stop:
                        break


            anim_output['Ground truth'] = torch.cat(anim_output['Ground truth'], dim=0).numpy()
            print(anim_output['Ground truth'].shape)
            anim_output['Ground truth'] = camera_to_world(anim_output['Ground truth'], R=cam['orientation'], t=cam['translation'])
            anim_output['Reconstruction'] = torch.cat(anim_output['Reconstruction'], dim=0).numpy()
            anim_output['Reconstruction'] = camera_to_world(anim_output['Reconstruction'], R=cam['orientation'],
                                                          t=cam['translation'])
            anim_diffusion_output['Diffusion process'] = camera_to_world(anim_diffusion_output['Diffusion process'].numpy(), R=cam['orientation'],
                                                          t=cam['translation'])


            anim_reverse_diffusion_output['Reverse diffusion process start'] = torch.stack(
                anim_reverse_diffusion_output['Reverse diffusion process start'], dim=0)
            anim_reverse_diffusion_output['Reverse diffusion process middle2'] = torch.stack(
                anim_reverse_diffusion_output['Reverse diffusion process middle2'], dim=0)
            anim_reverse_diffusion_output['Reverse diffusion process middle'] = torch.stack(
                anim_reverse_diffusion_output['Reverse diffusion process middle'], dim=0)
            anim_reverse_diffusion_output['Reverse diffusion process end'] = torch.stack(
                anim_reverse_diffusion_output['Reverse diffusion process end'], dim=0)
            anim_reverse_diffusion_input['Reverse diffusion process input 2D'] = torch.stack(
                anim_reverse_diffusion_input['Reverse diffusion process input 2D'], dim=0).numpy()
            anim_reverse_diffusion_input['Reverse diffusion process input 2D'][:,:,1] = -anim_reverse_diffusion_input['Reverse diffusion process input 2D'][:,:,1]

            anim_reverse_diffusion_output['Reverse diffusion process start'] = camera_to_world(
                anim_reverse_diffusion_output['Reverse diffusion process start'].numpy(), R=cam['orientation'],
                t=cam['translation'])
            anim_reverse_diffusion_output['Reverse diffusion process middle2'] = camera_to_world(
                anim_reverse_diffusion_output['Reverse diffusion process middle2'].numpy(), R=cam['orientation'],
                t=cam['translation'])
            anim_reverse_diffusion_output['Reverse diffusion process middle'] = camera_to_world(
                anim_reverse_diffusion_output['Reverse diffusion process middle'].numpy(), R=cam['orientation'],
                t=cam['translation'])
            anim_reverse_diffusion_output['Reverse diffusion process end'] = camera_to_world(
                anim_reverse_diffusion_output['Reverse diffusion process end'].numpy(), R=cam['orientation'],
                t=cam['translation'])

            input_keypoints = torch.cat(input_keypoints, dim=0).numpy()
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

            render_2d_figs(anim_reverse_diffusion_input['Reverse diffusion process input 2D'],
                           keypoints_metadata,
                           dataset.skeleton(),
                           os.path.join(viz_path,'2D_keypoints_'+args.viz_output),
                           size=args.viz_size)
            render_3d_figs(anim_reverse_diffusion_output['Reverse diffusion process start'],
                           dataset.skeleton(), cam['azimuth'],
                           os.path.join(viz_path, 'reverse_diffusion_start_' + args.viz_output),
                           size=args.viz_size)
            render_3d_figs(anim_reverse_diffusion_output['Reverse diffusion process middle2'],
                           dataset.skeleton(), cam['azimuth'],
                           os.path.join(viz_path, 'reverse_diffusion_middle2_' + args.viz_output),
                           size=args.viz_size)
            render_3d_figs(anim_reverse_diffusion_output['Reverse diffusion process middle'],
                           dataset.skeleton(), cam['azimuth'],
                           os.path.join(viz_path, 'reverse_diffusion_middle_' + args.viz_output),
                           size=args.viz_size)
            render_3d_figs(anim_reverse_diffusion_output['Reverse diffusion process end'],
                           dataset.skeleton(), cam['azimuth'],
                           os.path.join(viz_path, 'reverse_diffusion_end_'+args.viz_output),
                           size=args.viz_size)



