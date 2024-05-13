# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/run.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#

# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.nosiy_generators import ChunkedGenerator, ChunkedGenerator_3dhp


class load_Dataset(data.Dataset):
    def __init__(self, opt, dataset, root_path, split='train', action_filter=None, noise_std=0.0, joint_drop_rate=0):
        self.split = split
        self.dataset_name = opt.dataset
        self.keypoints_name = opt.keypoints
        self.noise_std = noise_std
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        if action_filter is None: # For train, val, test
            self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        else: # For final eval, only accept one action at a time
            assert isinstance(action_filter, list)
            self.action_filter = action_filter

        self.joint_drop_rate = joint_drop_rate
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = 0
        self.test_aug = opt.test_time_augmentation
        receptive_field = opt.number_of_frames
        self._w_mpjpe = dataset._w_mpjpe
        ###For 3D pose normalization###
        pos_3d_min = dataset._pos_3d_min
        pos_3d_max = dataset._pos_3d_max
        pos_3d_max_abs = np.absolute(pos_3d_max)
        pos_3d_min_abs = np.absolute(pos_3d_min)
        if pos_3d_max_abs >= pos_3d_min_abs:
            self.scale = pos_3d_max_abs
        else:
            self.scale = pos_3d_min_abs
        ###For 3D pose normalization###

        print('INFO: Receptive field: {} frames'.format(receptive_field))
        self.out_all = opt.out_all
        if opt.out_all:
            self.pad = 0
        else:
            self.pad = (receptive_field - 1) // 2  # Padding on each side

        print('Padding {} frames on each side'.format(self.pad))

        if self.action_filter is not None:
            print('Selected actions:', self.action_filter)

        if self.split == 'train':
            self.train = True
            self.keypoints = self.prepare_data(dataset, self.train_list)
            # cameras_train: cam['intrinsic']
            # poses_train: 3D pose in *camera space* without global offset in joint 1-16 (idx in positions_3d)
            # poses_train_2d: Normalize camera frame
            self.cameras_train, self.poses_train, self.poses_train_2d, out_frame_id = self.fetch(dataset, self.train_list,
                                                                                                                            subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, out_frame_id, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=False,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(len(self.generator.pairs)))
        elif self.split == 'test':
            self.train = False
            self.keypoints  = self.prepare_data(dataset, self.test_list)

            self.cameras_test, self.poses_test, self.poses_test_2d, out_frame_id = self.fetch(dataset, self.test_list,
                                                                                                                        subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size, self.cameras_test, self.poses_test,
                                              self.poses_test_2d, out_frame_id, self.stride, pad=self.pad,
                                              augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(len(self.generator.pairs)))
        elif self.split == 'val':
            self.train = True
            self.keypoints  = self.prepare_data(dataset, self.train_list)
            self.cameras_val, self.poses_val, self.poses_val_2d, out_frame_id = self.fetch(dataset, self.train_list,
                                                                                                                    subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size, self.cameras_val, self.poses_val,
                                              self.poses_val_2d, out_frame_id, self.stride, pad=self.pad,
                                              augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('INFO: Validating (train) on {} frames'.format(len(self.generator.pairs)))
        else:
            raise print('Unknown split {}'.format(self.split))

        if opt.out_all:
            self.get_batch = self.generator.get_batch_seq2seq
        else:
            self.get_batch = self.generator.get_batch_seq2frame

    def prepare_data(self, dataset, folder_list):
        print('Preparing data...')
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # pos_3d[:, 1:] -= pos_3d[:, :1]
                    pos_3d -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        print('Loading 2D detections...')
        keypoints = np.load(self.root_path + '/data_2d_' + self.dataset_name + '_' + self.keypoints_name + '.npz',
                            allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']
        self.num_joints = keypoints['metadata'].item()['num_joints']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(
            dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                if 'positions_3d' not in dataset[subject][action]:
                    continue

                for cam_idx in range(len(keypoints[subject][action])):

                    # We check for >= instead of == because some videos in H3.6M contain extra frames
                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

                assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

        for subject in folder_list:
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    # Normalize camera frame
                    cam = dataset.cameras()[subject][cam_idx]
                    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
                    if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

                    keypoints[subject][action][cam_idx] = kps

        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}
        out_frame_id = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]

                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d[(subject, action, i)] = poses_2d[i]
                    out_frame_id[(subject, action, i)] = np.arange(poses_2d[i].shape[0])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            # Downsample as requested
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_frame_id

    def norm_3d_pose(self, gt_3D):
        gt_3D_norm = gt_3D / self.scale
        return gt_3D_norm

    def reverse_norm_3d_pose(self, predicted_3d_pos):
        predicted_3d_pos = predicted_3d_pos * self.scale
        return predicted_3d_pos

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        if self.out_all:
            seq_name, start_3d, end_3d, start_target_3d, end_target_3d, flip, reverse = self.generator.pairs[index]
        else:
            seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
            start_target_3d = None

        cam, gt_3D, input_2D, target_mask, gt_3D_frame_id, input_2D_frame_id, action, subject, cam_ind = self.get_batch(seq_i=seq_name,
                                                                                                                                        start_3d=start_3d,
                                                                                                                                        end_3d=end_3d,
                                                                                                                                        start_target_3d=start_target_3d,
                                                                                                                                        flip=flip,
                                                                                                                                        reverse=reverse)
        gt_3D_norm = self.norm_3d_pose(gt_3D)
        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, input_2D_aug_frame_id, _, _, _ = self.get_batch(seq_i=seq_name,
                                                                                                          start_3d=start_3d, end_3d=end_3d,
                                                                                                          start_target_3d=start_target_3d,
                                                                                                          flip=True, reverse=False)

        else:
            input_2D_aug = []
            input_2D_aug_frame_id = []

        bb_box = np.array([0, 0, 1, 1])

        scale = 1.0
        if target_mask is None:
            target_mask = np.full(gt_3D.shape[0], True, dtype=bool)

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, input_2D.shape).astype('float32')
            input_2D = input_2D + noise
            if self.train == False and self.test_aug:
                noise = np.random.normal(0.0, self.noise_std, input_2D_aug.shape).astype('float32')
                input_2D_aug = input_2D_aug + noise

        if self.joint_drop_rate > 0:
            joint_drop_mask = np.repeat(
                np.random.binomial(1, 1 - self.joint_drop_rate, (input_2D.shape[0], input_2D.shape[1], 1)),
                input_2D.shape[2], axis=-1).astype('float32')
            input_2D = input_2D * joint_drop_mask
            if self.train == False and self.test_aug:
                joint_drop_mask = np.repeat(
                    np.random.binomial(1, 1 - self.joint_drop_rate, (input_2D_aug.shape[0], input_2D_aug.shape[1], 1)),
                    input_2D_aug.shape[2], axis=-1).astype('float32')
                input_2D_aug = input_2D_aug * joint_drop_mask

        return cam, gt_3D, gt_3D_norm, input_2D, input_2D_aug, target_mask, gt_3D_frame_id, input_2D_frame_id, input_2D_aug_frame_id, action, subject, scale, bb_box, cam_ind

class load_Dataset_3dhp(data.Dataset):
    def __init__(self, opt, dataset, pos_3d_min, pos_3d_max, split='train', noise_std=0.0, joint_drop_rate=0, seq_filter=None):
        self.split = split
        self.dataset_name = opt.dataset
        self.keypoints_name = opt.keypoints
        self.noise_std = noise_std

        self.joint_drop_rate = joint_drop_rate
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.test_aug = opt.test_time_augmentation
        receptive_field = opt.number_of_frames
        ###For 3D pose normalization###
        pos_3d_max_abs = np.absolute(pos_3d_max)
        pos_3d_min_abs = np.absolute(pos_3d_min)
        if pos_3d_max_abs >= pos_3d_min_abs:
            self.scale = pos_3d_max_abs
        else:
            self.scale = pos_3d_min_abs
        ###For 3D pose normalization###

        print('INFO: Receptive field: {} frames'.format(receptive_field))
        self.out_all = opt.out_all
        if opt.out_all:
            self.pad = 0
        else:
            self.pad = (receptive_field - 1) // 2  # Padding on each side

        print('Padding {} frames on each side'.format(self.pad))

        self.kps_left, self.kps_right = dataset.kps_left, dataset.kps_right
        self.joints_left, self.joints_right = dataset.joints_left, dataset.joints_right
        assert seq_filter is None or seq_filter in dataset.poses_3d.keys()
        if seq_filter is None:
            self.poses_3d = dataset.poses_3d
            self.poses_2d = dataset.poses_2d
            self.valid_frame = dataset.valid_frame
        else:
            self.poses_3d = {}
            self.poses_2d = {}
            self.valid_frame = {}
            self.poses_3d[seq_filter] = dataset.poses_3d[seq_filter]
            self.poses_2d[seq_filter] = dataset.poses_2d[seq_filter]
            self.valid_frame[seq_filter] = dataset.valid_frame[seq_filter]

        if self.split == 'train':
            self.train = True
            self.generator = ChunkedGenerator_3dhp(opt.batch_size, None, self.poses_3d,
                                                   self.poses_2d, self.stride, pad=self.pad,
                                                   augment=opt.data_augmentation, reverse_aug=False,
                                                   kps_left=self.kps_left, kps_right=self.kps_right,
                                                   joints_left=self.joints_left, joints_right=self.joints_right,
                                                   out_all=opt.out_all,
                                                   split=self.split)
            print('INFO: Training on {} frames'.format(len(self.generator.pairs)))
        elif self.split == 'test':
            self.train = False
            self.generator = ChunkedGenerator_3dhp(opt.batch_size, None, self.poses_3d,
                                                   self.poses_2d, self.stride, pad=self.pad,
                                                   augment=False,
                                                   kps_left=self.kps_left, kps_right=self.kps_right,
                                                   joints_left=self.joints_left, joints_right=self.joints_right,
                                                   out_all=opt.out_all,
                                                   valid_frame=self.valid_frame, split=self.split)
            print('INFO: Testing on {} frames'.format(len(self.generator.pairs)))
        elif self.split == 'val':
            self.train = True
            self.generator = ChunkedGenerator_3dhp(opt.batch_size, None, self.poses_3d,
                                                   self.poses_2d, self.stride, pad=self.pad,
                                                   augment=False, reverse_aug=False,
                                                   kps_left=self.kps_left, kps_right=self.kps_right,
                                                   joints_left=self.joints_left, joints_right=self.joints_right,
                                                   out_all=opt.out_all)
            print('INFO: Validating (train) on {} frames'.format(len(self.generator.pairs)))
        else:
            raise print('Unknown split {}'.format(self.split))

        if opt.out_all:
            self.get_batch = self.generator.get_batch_seq2seq
        else:
            self.get_batch = self.generator.get_batch_seq2frame

    def norm_3d_pose(self, gt_3D):
        gt_3D_norm = gt_3D / self.scale
        return gt_3D_norm

    def reverse_norm_3d_pose(self, predicted_3d_pos):
        predicted_3d_pos = predicted_3d_pos * self.scale
        return predicted_3d_pos

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        if self.out_all:
            seq_name, start_3d, end_3d, start_target_3d, end_target_3d, flip, reverse = self.generator.pairs[index]
        else:
            seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
            start_target_3d = None

        cam, gt_3D, input_2D, target_mask, seq, subject, cam_ind = self.get_batch(seq_i=seq_name,
                                                                                  start_3d=start_3d,
                                                                                  end_3d=end_3d,
                                                                                  start_target_3d=start_target_3d,
                                                                                  flip=flip,
                                                                                  reverse=reverse)
        gt_3D_norm = self.norm_3d_pose(gt_3D)
        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _, _, _ = self.get_batch(seq_i=seq_name,
                                                            start_3d=start_3d, end_3d=end_3d,
                                                            start_target_3d=start_target_3d,
                                                            flip=True, reverse=False)

        else:
            input_2D_aug = []

        if subject is None:
            subject = []

        if cam_ind is None:
            cam_ind = []

        bb_box = np.array([0, 0, 1, 1])

        scale = 1.0
        if target_mask is None:
            target_mask = np.full(gt_3D.shape[0], True, dtype=bool)

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, input_2D.shape).astype('float32')
            input_2D = input_2D + noise
            if self.train == False and self.test_aug:
                noise = np.random.normal(0.0, self.noise_std, input_2D_aug.shape).astype('float32')
                input_2D_aug = input_2D_aug + noise

        if self.joint_drop_rate > 0:
            joint_drop_mask = np.repeat(
                np.random.binomial(1, 1 - self.joint_drop_rate, (input_2D.shape[0], input_2D.shape[1], 1)),
                input_2D.shape[2], axis=-1).astype('float32')
            input_2D = input_2D * joint_drop_mask
            if self.train == False and self.test_aug:
                joint_drop_mask = np.repeat(
                    np.random.binomial(1, 1 - self.joint_drop_rate,
                                       (input_2D_aug.shape[0], input_2D_aug.shape[1], 1)),
                    input_2D_aug.shape[2], axis=-1).astype('float32')
                input_2D_aug = input_2D_aug * joint_drop_mask

        return cam, gt_3D, gt_3D_norm, input_2D, input_2D_aug, target_mask, seq, subject, scale, bb_box, cam_ind