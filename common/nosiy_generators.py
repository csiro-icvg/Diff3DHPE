# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/common/generators.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import numpy as np


class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, frame_id,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        if out_all:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_seq_frame = poses_2d[key].shape[0]
                n_chunks = (n_seq_frame + chunk_length - 1) // chunk_length
                bounds_continuous_chunk = np.arange(n_chunks) * chunk_length
                augment_vector = np.full(n_chunks, False, dtype=bool)
                reverse_augment_vector = np.full(n_chunks, False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 3]), (n_chunks, 1))
                start_index_last_chunk = n_seq_frame - chunk_length
                end_index_last_chunk = n_seq_frame
                target_offset = start_index_last_chunk - bounds_continuous_chunk[-1]
                start_index_last_chunk_target = start_index_last_chunk + target_offset
                end_index_last_chunk_target = end_index_last_chunk
                start_index_chunk = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk)
                end_index_chunk = np.append(bounds_continuous_chunk[1:], end_index_last_chunk)
                start_index_chunk_target = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk_target)
                end_index_chunk_target = np.append(bounds_continuous_chunk[1:], end_index_last_chunk_target)
                pairs += list(zip(keys,
                                  start_index_chunk, end_index_chunk,
                                  start_index_chunk_target, end_index_chunk_target,
                                  augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys,
                                      start_index_chunk, end_index_chunk,
                                      start_index_chunk_target, end_index_chunk_target,
                                      augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
                self.batch_3d_frame_id = np.empty((batch_size, chunk_length))

            self.batch_2d = np.empty(
                (batch_size, chunk_length, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))
            self.batch_2d_frame_id = np.empty((batch_size, chunk_length))

        else:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks + 1) * chunk_length - offset
                augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 3]), (len(bounds - 1), 1))
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
                self.batch_3d_frame_id = np.empty((batch_size, chunk_length))

            self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))
            self.batch_2d_frame_id = np.empty((batch_size, chunk_length + 2 * pad))



        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.chunk_length =chunk_length
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.frame_id = frame_id

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch_seq2frame(self, seq_i, start_3d, end_3d, flip, reverse, **kwargs):
        target_mask = None
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        seq_2d_frame_id = self.frame_id[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
            self.batch_2d_frame_id = np.pad(seq_2d_frame_id[low_2d:high_2d], ((pad_left_2d, pad_right_2d)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]
            self.batch_2d_frame_id = seq_2d_frame_id[low_2d:high_2d]

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()
            self.batch_2d_frame_id = self.batch_2d_frame_id[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            seq_3d_frame_id = self.frame_id[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                self.batch_3d_frame_id = np.pad(seq_3d_frame_id[low_3d:high_3d],
                                                ((pad_left_3d, pad_right_3d)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                self.batch_3d_frame_id = seq_3d_frame_id[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                self.batch_3d_frame_id = self.batch_3d_frame_id[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), None, target_mask, \
                   None, self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), None, target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy(), None, \
                   None, self.batch_2d_frame_id.copy(), target_mask, \
                   action, subject, int(cam_index)
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), None, target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)

    def get_batch_seq2seq(self, seq_i, start_3d, end_3d, start_target_3d, flip, reverse):
        target_mask = None
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        seq_2d_frame_id = self.frame_id[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
            self.batch_2d_frame_id = np.pad(seq_2d_frame_id[low_2d:high_2d], ((pad_left_2d, pad_right_2d)), 'edge')
        # Should only apply to out_all case
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]
            self.batch_2d_frame_id = seq_2d_frame_id[low_2d:high_2d]
            target_mask = np.full(self.chunk_length, True, dtype=bool)
            n_unused_frame = start_3d - start_target_3d
            assert n_unused_frame >= 0
            if n_unused_frame > 0:
                target_mask[:n_unused_frame] = False

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]

        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()
            self.batch_2d_frame_id = self.batch_2d_frame_id[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            seq_3d_frame_id = self.frame_id[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                self.batch_3d_frame_id = np.pad(seq_3d_frame_id[low_3d:high_3d],
                                                ((pad_left_3d, pad_right_3d)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                self.batch_3d_frame_id = seq_3d_frame_id[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                self.batch_3d_frame_id = self.batch_3d_frame_id[::-1].copy()
                target_mask = target_mask[::-1]

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), target_mask, \
                   None, self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy(), \
                   None, self.batch_2d_frame_id.copy(), target_mask, \
                   action, subject, int(cam_index)
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)


class ChunkedGenerator_3dhp:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False, valid_frame=None, split='train'):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
        assert valid_frame is None or len(valid_frame) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0
        self.split = split

        if out_all:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_seq_frame = poses_2d[key].shape[0]
                n_chunks = (n_seq_frame + chunk_length - 1) // chunk_length
                # offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds_continuous_chunk = np.arange(n_chunks) * chunk_length
                augment_vector = np.full(n_chunks, False, dtype=bool)
                reverse_augment_vector = np.full(n_chunks, False, dtype=bool)
                if self.split == 'train' or self.split == 'val':
                    keys = np.tile(np.array(key).reshape([1, 3]), (n_chunks, 1))
                else:
                    keys = np.tile(np.array(key).reshape([1]), (n_chunks, 1))

                start_index_last_chunk = n_seq_frame - chunk_length
                end_index_last_chunk = n_seq_frame
                target_offset = start_index_last_chunk - bounds_continuous_chunk[-1]
                start_index_last_chunk_target = start_index_last_chunk + target_offset
                end_index_last_chunk_target = end_index_last_chunk
                start_index_chunk = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk)
                end_index_chunk = np.append(bounds_continuous_chunk[1:], end_index_last_chunk)
                start_index_chunk_target = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk_target)
                end_index_chunk_target = np.append(bounds_continuous_chunk[1:], end_index_last_chunk_target)
                pairs += list(zip(keys,
                                  start_index_chunk, end_index_chunk,
                                  start_index_chunk_target, end_index_chunk_target,
                                  augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys,
                                      start_index_chunk, end_index_chunk,
                                      start_index_chunk_target, end_index_chunk_target,
                                      augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))

            self.batch_2d = np.empty(
                (batch_size, chunk_length, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))

        else:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks + 1) * chunk_length - offset
                augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                if self.split == 'train' or self.split == 'val':
                    keys = np.tile(np.array(key).reshape([1, 3]), (len(bounds - 1), 1))
                else:
                    keys = np.tile(np.array(key).reshape([1]), (len(bounds - 1), 1))

                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))

            self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.chunk_length =chunk_length
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.valid_frame = valid_frame

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch_seq2frame(self, seq_i, start_3d, end_3d, flip, reverse, **kwargs):
        target_mask = None
        seq_valid_frame = None
        batch_valid_frame = None
        if self.split == 'train' or self.split == 'val':
            subject, seq, cam_index = seq_i
            seq_name = (subject, seq, cam_index)
        else:
            subject, seq, cam_index = None, seq_i[0], None
            seq_name = seq

        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.valid_frame is not None:
                seq_valid_frame = self.valid_frame[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                if self.valid_frame is not None:
                    batch_valid_frame = np.pad(seq_valid_frame[low_3d:high_3d],
                                              ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')

            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                if self.valid_frame is not None:
                    batch_valid_frame = seq_valid_frame[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                if self.valid_frame is not None:
                    batch_valid_frame = batch_valid_frame[::-1]

        if batch_valid_frame is not None:
            target_mask = batch_valid_frame.astype(bool)

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index

    def get_batch_seq2seq(self, seq_i, start_3d, end_3d, start_target_3d, flip, reverse):
        target_mask = None
        seq_valid_frame = None
        batch_valid_frame = None
        if self.split == 'train' or self.split == 'val':
            subject, seq, cam_index = seq_i
            seq_name = (subject, seq, cam_index)
        else:
            subject, seq, cam_index = None, seq_i[0], None
            seq_name = seq

        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        # Should only apply to out_all case
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]
            target_mask = np.full(self.chunk_length, True, dtype=bool)
            n_unused_frame = start_3d - start_target_3d
            assert n_unused_frame >= 0
            if n_unused_frame > 0:
                target_mask[:n_unused_frame] = False

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.valid_frame is not None:
                seq_valid_frame = self.valid_frame[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                if self.valid_frame is not None:
                    batch_valid_frame = seq_valid_frame[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                target_mask = target_mask[::-1]
                if self.valid_frame is not None:
                    batch_valid_frame = batch_valid_frame[::-1]

        if self.valid_frame is not None:
            target_mask = target_mask & batch_valid_frame.astype(bool)

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   seq, subject, cam_index


class ChunkedGenerator_viz:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, frame_id,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0
        self.total_n_seq_frame = 0

        if out_all:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_seq_frame = poses_2d[key].shape[0]
                self.total_n_seq_frame += n_seq_frame
                n_chunks = (n_seq_frame + chunk_length - 1) // chunk_length
                # offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds_continuous_chunk = np.arange(n_chunks) * chunk_length
                augment_vector = np.full(n_chunks, False, dtype=bool)
                reverse_augment_vector = np.full(n_chunks, False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 3]), (n_chunks, 1))
                start_index_last_chunk = n_seq_frame - chunk_length
                end_index_last_chunk = n_seq_frame
                target_offset = start_index_last_chunk - bounds_continuous_chunk[-1]
                start_index_last_chunk_target = start_index_last_chunk + target_offset
                end_index_last_chunk_target = end_index_last_chunk
                start_index_chunk = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk)
                end_index_chunk = np.append(bounds_continuous_chunk[1:], end_index_last_chunk)
                start_index_chunk_target = np.append(bounds_continuous_chunk[:-1], start_index_last_chunk_target)
                end_index_chunk_target = np.append(bounds_continuous_chunk[1:], end_index_last_chunk_target)
                pairs += list(zip(keys,
                                  start_index_chunk, end_index_chunk,
                                  start_index_chunk_target, end_index_chunk_target,
                                  augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys,
                                      start_index_chunk, end_index_chunk,
                                      start_index_chunk_target, end_index_chunk_target,
                                      augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys,
                                          start_index_chunk, end_index_chunk,
                                          start_index_chunk_target, end_index_chunk_target,
                                          ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
                self.batch_3d_frame_id = np.empty((batch_size, chunk_length))

            self.batch_2d = np.empty(
                (batch_size, chunk_length, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))
            self.batch_2d_frame_id = np.empty((batch_size, chunk_length))

        else:
            for key in poses_2d.keys():
                assert poses_3d is None or poses_3d[key].shape[0] == poses_3d[key].shape[0]
                n_seq_frame = poses_2d[key].shape[0]
                self.total_n_seq_frame += n_seq_frame
                n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
                offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
                bounds = np.arange(n_chunks + 1) * chunk_length - offset
                augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
                keys = np.tile(np.array(key).reshape([1, 3]), (len(bounds - 1), 1))
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
                if augment:
                    if reverse_aug:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                    else:
                        pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))

                end_index = start_index + poses_3d[key].shape[0]
                self.saved_index[key] = [start_index, end_index]
                start_index = start_index + poses_3d[key].shape[0]

            if cameras is not None:
                self.batch_cam = np.empty((batch_size, cameras[key].shape[-1]))

            if poses_3d is not None:
                self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
                self.batch_3d_frame_id = np.empty((batch_size, chunk_length))

            self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-2], poses_2d[key].shape[-1]))
            self.batch_2d_frame_id = np.empty((batch_size, chunk_length + 2 * pad))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.chunk_length =chunk_length
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.frame_id = frame_id

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch_seq2frame(self, seq_i, start_3d, end_3d, flip, reverse, **kwargs):
        target_mask = None
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        seq_2d_frame_id = self.frame_id[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
            self.batch_2d_frame_id = np.pad(seq_2d_frame_id[low_2d:high_2d], ((pad_left_2d, pad_right_2d)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]
            self.batch_2d_frame_id = seq_2d_frame_id[low_2d:high_2d]

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()
            self.batch_2d_frame_id = self.batch_2d_frame_id[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            seq_3d_frame_id = self.frame_id[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                self.batch_3d_frame_id = np.pad(seq_3d_frame_id[low_3d:high_3d],
                                                ((pad_left_3d, pad_right_3d)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                self.batch_3d_frame_id = seq_3d_frame_id[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                self.batch_3d_frame_id = self.batch_3d_frame_id[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), target_mask, \
                   None, self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy(), \
                   None, self.batch_2d_frame_id.copy(), target_mask, \
                   action, subject, int(cam_index)
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)

    def get_batch_seq2seq(self, seq_i, start_3d, end_3d, start_target_3d, flip, reverse):
        target_mask = None
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        seq_2d_frame_id = self.frame_id[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
            self.batch_2d_frame_id = np.pad(seq_2d_frame_id[low_2d:high_2d], ((pad_left_2d, pad_right_2d)), 'edge')
        # Should only apply to out_all case
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]
            self.batch_2d_frame_id = seq_2d_frame_id[low_2d:high_2d]
            target_mask = np.full(self.chunk_length, True, dtype=bool)
            n_unused_frame = start_3d - start_target_3d
            assert n_unused_frame >= 0
            if n_unused_frame > 0:
                target_mask[:n_unused_frame] = False

        if flip:
            self.batch_2d[:, :, 0] *= -1
            self.batch_2d[:, self.kps_left + self.kps_right] = self.batch_2d[:,
                                                               self.kps_right + self.kps_left]

        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()
            self.batch_2d_frame_id = self.batch_2d_frame_id[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            seq_3d_frame_id = self.frame_id[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                self.batch_3d_frame_id = np.pad(seq_3d_frame_id[low_3d:high_3d],
                                                ((pad_left_3d, pad_right_3d)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
                self.batch_3d_frame_id = seq_3d_frame_id[low_3d:high_3d]

            if flip:
                self.batch_3d[:, :, 0] *= -1
                self.batch_3d[:, self.joints_left + self.joints_right] = \
                    self.batch_3d[:, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()
                self.batch_3d_frame_id = self.batch_3d_frame_id[::-1].copy()
                target_mask = target_mask[::-1]

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, \
                   None, self.batch_2d.copy(), target_mask, \
                   None, self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, \
                   None, self.batch_2d.copy().copy(), \
                   None, self.batch_2d_frame_id.copy(), target_mask, \
                   action, subject, int(cam_index)
        else:
            return self.batch_cam, \
                   self.batch_3d.copy(), self.batch_2d.copy(), target_mask, \
                   self.batch_3d_frame_id.copy(), self.batch_2d_frame_id.copy(), \
                   action, subject, int(cam_index)






