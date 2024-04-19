# This file is modified from https://github.com/facebookresearch/VideoPose3D/blob/main/common/loss.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Used under the CC-BY-4.0 license: https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F

def mpjpe(predicted, target, reduce='mean'):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if reduce == 'mean':
        out = torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    elif reduce == 'none':
        out = torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1), dim=-1)
    else:
        raise ValueError('Unknown reduce method: {}'.format(reduce))
    return out
    
def weighted_mpjpe(predicted, target, w, reduce='mean'):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    if reduce == 'mean':
        out = torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))
    elif reduce == 'none':
        out = torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1), dim=-1)
    else:
        raise ValueError('Unknown reduce method: {}'.format(reduce))

    return out

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def diff(input, dim=0):
    assert len(input.shape) == 4
    if dim == 0:
        out = input[1:,:,:,:] - input[:-1,:,:,:]
    elif dim == 1:
        out = input[:,1:,:,:] - input[:,:-1,:,:]
    else:
        raise print('Unsupport dim {}'.format(dim))

    return out

def mean_velocity_error_train(predicted, target, axis=0, reduce='mean'):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = diff(predicted, dim=axis)
    velocity_target = diff(target, dim=axis)

    if reduce == 'mean':
        out = torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape) - 1))
    elif reduce == 'none':
        out = torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape) - 1), dim=-1)
    else:
        raise ValueError('Unknown reduce method: {}'.format(reduce))

    return out

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def l1Loss(predicted, target, reduce='mean'):
    assert predicted.shape == target.shape
    out = F.l1_loss(predicted, target, reduction=reduce)

    return out

def l2Loss(predicted, target, reduce='mean'):
    assert predicted.shape == target.shape
    out = F.mse_loss(predicted, target, reduction=reduce)

    return out