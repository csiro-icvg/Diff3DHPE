# This file is modified from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# Used under the MIT license: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/LICENSE
#
# All modifications by CSIRO:
# Copyright (c) 2024-present, CSIRO
# All rights reserved.
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import math
from functools import partial

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t


def pad_with(vector, pad_width, iaxis, kwargs):
    if pad_width[0] == 0 and pad_width[1] == 0:
        pass
    else:
        tau = kwargs.get('tau', 1)
        num = vector.shape[0]
        centre_value = vector[(num - 1) // 2]
        vector[:pad_width[0]] = np.linspace(centre_value - tau, centre_value - 1, tau)
        vector[-pad_width[1]:] = np.linspace(centre_value + 1, centre_value + tau, tau)


### Schedules for the T timesteps ###
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def log_cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    # x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    x = torch.logspace(0, 2, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / 1e-1 / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

### Schedules for the T timesteps ###


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model,
                 timesteps=100,
                 sampling_timesteps=20,
                 loss_type='l1',
                 conditional=True,
                 clip_denoised=False,
                 beta_schedule='cosine',
                 p2_loss_weight_gamma=0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
                 p2_loss_weight_k=1,
                 ddim_sampling_eta=0.,
                 clipLoss=False
                 ):
        super().__init__()
        self.model = model
        self.conditional = conditional
        self.clip_denoised = clip_denoised
        self.clipLoss = clipLoss

        # define beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'logcosine':
            betas = log_cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(F.pad(alphas_cumprod, (1, 0), value=1.))
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps,
                                          timesteps)  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def add_temporal_condition(self, source, tau):
        b, f, p, c = source.shape
        max_ind = f - 1
        pad_ind = np.expand_dims(np.linspace(0, f - 1, f), axis=-1)
        new_pad_ind = np.pad(pad_ind, ((0, 0), (tau, tau)), pad_with, tau=tau)
        new_pad_ind[new_pad_ind < 0] = 0
        new_pad_ind[new_pad_ind > max_ind] = max_ind

        centre_ind = (new_pad_ind.shape[1] - 1) // 2
        new_pad_ind = np.concatenate((new_pad_ind[:, :centre_ind], new_pad_ind[:, centre_ind + 1:]), axis=1).astype(int)
        new_pad_ind = np.resize(new_pad_ind, f * 2 * tau)

        source = rearrange(source, 'b f p c -> f (b p c)', b=b)
        conditions = source[new_pad_ind, :]
        conditions = rearrange(conditions, '(f pad) (b p c) -> b f p (c pad)', b=b, f=f, p=p, pad=2 * tau, c=c)
        return conditions

    def predict_start_from_noise(self, x_t, t, noise):
        return (x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise) / self.sqrt_alphas_cumprod[t]

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size).to(x.device)
        if condition_x is not None:
            x_start = self.predict_start_from_noise(
                x, t=t, noise=self.model.forward_denoise(torch.cat([condition_x, x], dim=-1), noise_level))
        else:
            x_start = self.predict_start_from_noise(
                x, t=t, noise=self.model.forward_denoise(x, noise_level))

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    # Denoise
    @torch.no_grad()
    def p_sample_loop(self, x_in, target_shape):
        device = self.betas.device
        if not self.conditional:
            noisy_3D_pose = torch.randn(target_shape, device=device)
            for i in reversed(range(0, self.num_timesteps)):
                noisy_3D_pose = self.p_sample(noisy_3D_pose, i)

        else:
            x = x_in
            noisy_3D_pose = torch.randn(target_shape, device=device)
            for i in reversed(range(0, self.num_timesteps)):
                noisy_3D_pose = self.p_sample(noisy_3D_pose, i, condition_x=x)

        clean_3D_pose = noisy_3D_pose
        return clean_3D_pose

    @torch.no_grad()
    def ddim_sample(self, x, t, condition_x=None):
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if self.clip_denoised else identity
        batch_size = x.shape[0]
        time = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
        x_start = self.model.forward_denoise(torch.cat([condition_x, x], dim=-1), time)
        x_start = maybe_clip(x_start)

        return x_start


    # DDIM denoise
    @torch.no_grad()
    def ddim_sample_loop(self, x_in, target_shape):
        f = x_in.shape[1]
        device = self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        noisy_3D_pose = torch.randn(target_shape, device=device)

        for time, time_next in time_pairs:

            x_condition = x_in

            x_start = self.ddim_sample(noisy_3D_pose.repeat(1, f, 1, 1), time, condition_x=x_condition)

            if time_next < 0:
                noisy_3D_pose = x_start
                continue

            alpha = self.alphas_cumprod[time].view(-1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[time_next].view(-1, 1, 1, 1)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(noisy_3D_pose)

            noisy_3D_pose = x_start * alpha_next.sqrt() + \
                            c * ((noisy_3D_pose - alpha * x_start) / self.sqrt_one_minus_alphas_cumprod[time].view(-1, 1, 1, 1)) + \
                            sigma * noise

        clean_3D_pose = noisy_3D_pose
        return clean_3D_pose

    # For visualization
    @torch.no_grad()
    def ddim_sample_loop_ouput_reverse_diffusion(self, x_in, target_shape):
        f = x_in.shape[1]
        device = self.betas.device
        x_reverse_diffusion = []
        x_start_est = []
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        noisy_3D_pose = torch.randn(target_shape, device=device)
        x_reverse_diffusion.append(noisy_3D_pose)

        for time, time_next in time_pairs:

            x_condition = x_in

            x_start = self.ddim_sample(noisy_3D_pose.repeat(1, f, 1, 1), time, condition_x=x_condition)
            x_start_est.append(x_start)

            if time_next < 0:
                noisy_3D_pose = x_start
                x_reverse_diffusion.append(noisy_3D_pose)
                continue

            alpha = self.alphas_cumprod[time].view(-1, 1, 1, 1)
            alpha_next = self.alphas_cumprod[time_next].view(-1, 1, 1, 1)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(noisy_3D_pose)

            noisy_3D_pose = x_start * alpha_next.sqrt() + \
                            c * ((noisy_3D_pose - alpha * x_start) / self.sqrt_one_minus_alphas_cumprod[time].view(-1, 1, 1, 1)) + \
                            sigma * noise
            x_reverse_diffusion.append(noisy_3D_pose)

        clean_3D_pose = noisy_3D_pose
        x_reverse_diffusion = torch.stack(x_reverse_diffusion, dim=-1)
        x_start_est = torch.stack(x_start_est, dim=-1)
        return clean_3D_pose, x_reverse_diffusion, x_start_est

    # Denoise and estimate 3D pose
    def forward_estimate_pose(self, x, target_shape, output_reverse_diffusion_3d=False):
        with torch.no_grad():
            if output_reverse_diffusion_3d:
                x = self.ddim_sample_loop_ouput_reverse_diffusion(x, target_shape)
            else:
                x = self.ddim_sample_loop(x, target_shape)

        return x


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # random gama
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    @torch.no_grad()
    def get_noisy_pose(self, x_start, num_sample, noise=None):
        b, f, p, c = x_start.shape
        t_list = list(np.arange(0, self.num_timesteps, self.num_timesteps // num_sample))
        x_diffusion = [x_start]
        for t_sample in t_list:
            t = torch.FloatTensor([t_sample]).long().repeat(b).to(x_start.device)
            x_noisy = self.q_sample(
                x_start=x_start, t=t, noise=noise)
            x_diffusion.append(x_noisy)

        x_diffusion = torch.stack(x_diffusion, dim=-1)
        return x_diffusion

    # Estimate noise
    def p_losses(self, x_start, pose_2d, noise=None):
        b, f, p, c = pose_2d.shape

        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        target = x_start
        x_noisy = self.q_sample(
            x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            model_out = self.model.forward_denoise(x_noisy.repeat(1, f, 1, 1), t)
        else:
            x_condition = pose_2d

            model_out = self.model.forward_denoise(
                torch.cat([x_condition, x_noisy.repeat(1, f, 1, 1)], dim=-1), t)

        # 1 + k_t
        loss_coef = (1.0 + self.alphas_cumprod[t].view(-1, 1, 1, 1) /
                     self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1))
        if self.clipLoss:
            loss_coef = torch.clamp(loss_coef, max=3.0)
        else:
            pass

        loss = self.loss_fn(model_out, target, reduction='none') * loss_coef
        return loss

    def forward(self, clean_3d_pose, noisy_2d_pose, noise=None, output_reverse_diffusion_3d=False, output_loss=True,
                repeat_n=1):
        if self.training:
            loss_pose = self.p_losses(clean_3d_pose, noisy_2d_pose, noise)
            predicted_3d_pos = None
            return loss_pose, predicted_3d_pos
        else:
            if output_loss:
                loss_pose = self.p_losses(clean_3d_pose, noisy_2d_pose, noise)
            else:
                loss_pose = None

            b, f, p, c = clean_3d_pose.shape
            noisy_2d_pose = noisy_2d_pose.repeat(repeat_n, 1, 1, 1)
            target_shape = list(clean_3d_pose.shape)
            target_shape[0] = target_shape[0] * repeat_n
            if output_reverse_diffusion_3d:
                predicted_3d_pos, reverse_diffusion_3d_pos, start_3d_pos_est = self.forward_estimate_pose(noisy_2d_pose,
                                                                                                          target_shape=target_shape,
                                                                                                          output_reverse_diffusion_3d=output_reverse_diffusion_3d)
                predicted_3d_pos = torch.mean(predicted_3d_pos.view(repeat_n, b, f, p, -1), dim=0, keepdim=True).squeeze(0)
                return loss_pose, predicted_3d_pos, reverse_diffusion_3d_pos, start_3d_pos_est

            else:
                predicted_3d_pos = self.forward_estimate_pose(noisy_2d_pose,
                                                              target_shape=target_shape,
                                                              output_reverse_diffusion_3d=output_reverse_diffusion_3d)
                predicted_3d_pos = torch.mean(predicted_3d_pos.view(repeat_n, b, f, p, -1), dim=0, keepdim=True).squeeze(0)
                return loss_pose, predicted_3d_pos