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
from common.nosiy_generators import ChunkedGenerator_viz as ChunkedGenerator


class load_Dataset(data.Dataset):
    def __init__(self, opt, dataset, root_path, sbj_list, action_filter, camera_filter, noise_std=0.0, joint_drop_rate=0.0):
        self.dataset_name = opt.dataset
        self.keypoints_name = opt.keypoints
        self.root_path = root_path
        self.sbj_list = sbj_list
        self.action_filter = action_filter
        self.camera_filter = camera_filter

        self.noise_std = noise_std
        self.joint_drop_rate = joint_drop_rate

        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = 0
        self.test_aug = opt.test_time_augmentation
        receptive_field = opt.number_of_frames
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

        print('Selected subjects:', self.sbj_list)
        if self.action_filter is not None:
            print('Selected actions:', self.action_filter)
        else:
            print('Selected all actions')

        if self.camera_filter is not None:
            print('Selected camera id:', self.camera_filter)
        else:
            print('Selected all cameras')


        self.train = False
        self.keypoints  = self.prepare_data(dataset, self.sbj_list)

        self.cameras_viz, self.poses_viz, self.poses_viz_2d, out_frame_id = self.fetch(dataset, self.sbj_list, subset=self.subset)
        self.generator = ChunkedGenerator(opt.batch_size, self.cameras_viz, self.poses_viz,
                                          self.poses_viz_2d, out_frame_id, self.stride, pad=self.pad,
                                          augment=False, kps_left=self.kps_left,
                                          kps_right=self.kps_right, joints_left=self.joints_left,
                                          joints_right=self.joints_right, out_all=opt.out_all)
        self.key_index = self.generator.saved_index
        print('INFO: Visualizing on {} frames'.format(self.generator.total_n_seq_frame))

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
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        print('Loading 2D detections...')
        keypoints = np.load(self.root_path + '/data_2d_' + self.dataset_name + '_' + self.keypoints_name + '.npz',
                            allow_pickle=True)

        self.keypoints_metadata = keypoints['metadata'].item()
        keypoints_symmetry = self.keypoints_metadata['keypoints_symmetry']
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

                for i in self.camera_filter: # Iterate across selected cameras
                    out_poses_2d[(subject, action, i)] = poses_2d[i]
                    out_frame_id[(subject, action, i)] = np.arange(poses_2d[i].shape[0])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i in self.camera_filter:
                        cam = cams[i]
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in self.camera_filter: # Iterate across selected cameras
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
        trajectory = gt_3D[:, :1].copy()
        gt_3D -= trajectory
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

        scale = float(1.0)
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

        return cam, trajectory, gt_3D, gt_3D_norm, input_2D, input_2D_aug, target_mask, gt_3D_frame_id, input_2D_frame_id, input_2D_aug_frame_id, action, subject, scale, bb_box, cam_ind