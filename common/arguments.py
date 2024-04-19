# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/common/arguments.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")

    # General arguments
    parser.add_argument('--seed', type=int,
                        help='Please give a value for seed')
    parser.add_argument('-d', '--dataset', type=str, metavar='NAME', help='target dataset. h36m/3dhp')
    parser.add_argument('--model', type=str, metavar='NAME',
                        help='Model name')
    parser.add_argument('-k', '--keypoints', type=str, metavar='NAME', help='2D keypoints to use. gt/cpn')
    parser.add_argument('--train_extra_noise_std', type=float, default=0.0,
                        help='std of extra Gaussian noise added to 2D pose in training set. Only for ablation study.')
    parser.add_argument('--train_val_extra_noise_std', type=float, default=0.0,
                        help='std of extra Gaussian noise added to 2D pose in train_val set. Only for ablation study.')
    parser.add_argument('--test_extra_noise_std', type=float, default=0.0,
                        help='std of extra Gaussian noise added to 2D pose in test set. Only for ablation study.')
    parser.add_argument('--train_joint_drop', type=float, default=0.0,
                        help='drop rate of joints in training set. Only for ablation study.')
    parser.add_argument('--train_val_joint_drop', type=float, default=0.0,
                        help='drop rate of joints in train_val set. Only for ablation study.')
    parser.add_argument('--test_joint_drop', type=float, default=0.0,
                        help='drop rate of joints in test set. Only for ablation study.')
    parser.add_argument('-str', '--subjects-train', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=40, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--pretrained_model', default=None, type=str, metavar='PATH',
                        help='pretrained model directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')
    parser.add_argument('--gpu_id', nargs='+', help="gpu id separated by comma")
    parser.add_argument('--out_all', help='Set True to use all frames as the target')
    parser.add_argument('--repeat_n', type=int,
                        help='Please give a value for repeat_n')


    # Model arguments
    parser.add_argument('-s', '--stride', type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('--batch-size-eval', type=int, metavar='N', help='batch size in terms of predicted frames when eval')
    parser.add_argument('-drop', '--dropout', default=0., type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('--patience', type=int, metavar='N', help='Number of patience for early stop')
    parser.add_argument('--timesteps', type=int, help='timesteps of diffusion')
    parser.add_argument('--sampling_timesteps', type=int, help='sampling timesteps of DDIM')
    parser.add_argument('--data-augmentation',
                        help='Set True to use train-time flipping')
    parser.add_argument('--test-time-augmentation',
                        help='Set True to use test-time flipping')
    # parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS', help='filter widths separated by comma')
    parser.add_argument('-frame', '--number-of-frames', type=int, metavar='N',
                        help='how many frames used as input')
    parser.add_argument('--transformer_depth', type=int, metavar='N', help='depth of each transformer module')
    parser.add_argument('--clip_denoised', help="set True to clip x_start")
    parser.add_argument('--with_time_emb', help="set True to use time emb")
    parser.add_argument('--beta_schedule', default='cosine', type=str,
                        help='beta schedule for diffusion model')
    parser.add_argument('--embed_dim', type=int, metavar='N',
                        help='Number of embedding dim')
    parser.add_argument('--ddim_sampling_eta', default=0.0, type=float, metavar='FACTOR',
                        help='ddim sampling eta')
    parser.add_argument('--loss_type', default='mpjpe', type=str,
                        help='loss type (only for run_poseformer.py)')
    parser.add_argument('--max_time', default=48, type=float,
                        help='max training time (hours)')
    parser.add_argument('--clip_loss', help="set True to clip loss")

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--no-eval', action='store_true', help='disable epoch evaluation while training (small speed-up)')

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args