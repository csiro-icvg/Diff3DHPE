# This file is modified from https://github.com/paTRICK-swk/P-STMO/blob/main/common/load_data_3dhp_mae.py
# Used under the MIT license: https://github.com/paTRICK-swk/P-STMO/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.


import os

import torch.utils.data as data
import numpy as np

from common.camera import normalize_screen_coordinates

kps_left, kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
joints_left, joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
num_joints = 17

class BaseMPIINF3DHPDataset(data.Dataset):
    def __init__(self, path, subjects_list, train=True):
        data = np.load(path, allow_pickle=True)['data'].item()
        self.kps_left, self.kps_right = kps_left, kps_right
        self.joints_left, self.joints_right = joints_left, joints_right
        self.num_joints = num_joints
        self.subjects_list = subjects_list
        self.prepare_data(data, train)

    def prepare_data(self, data, train=True):
        print('Preparing data...')
        out_poses_3d = {}
        out_poses_2d = {}
        all_pos_3d_coo_centred = []

        if train == True:
            valid_frame = None
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d -= data_3d[:, 14:15]
                    data_3d = data_3d.astype('float32')
                    all_pos_3d_coo_centred.append(data_3d)
                    if subject_name in self.subjects_list:
                        out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                        data_2d = anim['data_2d']

                        data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                        data_2d = data_2d.astype('float32')
                        out_poses_2d[(subject_name, seq_name, cam)] = data_2d
                    else:
                        pass
        else:
            valid_frame = {}
            for seq in data.keys():

                anim = data[seq]

                data_3d = anim['data_3d']
                data_3d -= data_3d[:, 14:15]
                data_3d = data_3d.astype('float32')
                all_pos_3d_coo_centred.append(data_3d)
                if seq in self.subjects_list:
                    valid_frame[seq] = anim["valid"]
                    out_poses_3d[seq] = data_3d

                    data_2d = anim['data_2d']

                    if seq == "TS5" or seq == "TS6":
                        width = 1920
                        height = 1080
                    else:
                        width = 2048
                        height = 2048
                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                    data_2d = data_2d.astype('float32')
                    out_poses_2d[seq] = data_2d
                else:
                    pass

        self.poses_3d = out_poses_3d
        self.poses_2d = out_poses_2d
        self.valid_frame = valid_frame

        all_pos_3d_coo_centred = np.concatenate(all_pos_3d_coo_centred, axis=0)
        self.pos_3d_min = all_pos_3d_coo_centred.min()
        self.pos_3d_max = all_pos_3d_coo_centred.max()




class MPIINF3DHPDataset(data.Dataset):
    def __init__(self, opt, root_path='data'):
        train_list = opt.subjects_train.split(',')
        test_list = opt.subjects_test.split(',')
        path_train = os.path.join(root_path, 'data_train_3dhp.npz')
        path_test = os.path.join(root_path, 'data_test_3dhp.npz')
        train = BaseMPIINF3DHPDataset(path_train, train_list, train=True)
        test = BaseMPIINF3DHPDataset(path_test, test_list, train=False)
        pos_3d_min, pos_3d_max = self.get_min_max_pose_3d([train.pos_3d_min, test.pos_3d_min],
                                                          [train.pos_3d_max, test.pos_3d_max])
        self._pos_3d_min = pos_3d_min
        self._pos_3d_max = pos_3d_max
        self._train = train
        self._test = test
        self.kps_left, self.kps_right = kps_left, kps_right
        self.joints_left, self.joints_right = joints_left, joints_right
        self.num_joints = num_joints

    def get_min_max_pose_3d(self, min_list, max_list):
        out_min = np.inf
        out_max = -np.inf
        for min in min_list:
            if min < out_min:
                out_min = min

        for max in max_list:
            if max > out_max:
                out_max = max

        return out_min, out_max
