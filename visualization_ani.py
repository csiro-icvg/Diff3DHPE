import os
import errno
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
ffprobe_path = '/YOUR/PATH/TO/bin/ffprobe' # Must be set according to your environment
ffmpeg_path = '/usr/bin/ffmpeg' # Must be set according to your environment
rcParams['animation.ffmpeg_path'] = ffmpeg_path
from matplotlib.animation import FuncAnimation, PillowWriter, writers
import subprocess as sp
import torch.nn as nn

from common.camera import *

from common.nets.load_net import HPE_model
from common.conditional_diffusion_ddim_normal_directPredict_variableLoss_both_crossFrames import GaussianDiffusion
from common.loss import *
from common.utils import *

from data.load_noisy_data_viz import load_Dataset

def get_resolution(filename):
    command = [ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stderr=sp.PIPE, stdout=sp.PIPE, shell=False) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = [ffprobe_path, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stderr=sp.PIPE, stdout=sp.PIPE, shell=False) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = [ffmpeg_path,
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']
    i = 0
    pipe = sp.Popen(command, stderr=sp.PIPE, stdout=sp.PIPE, shell=False)
    while True:
        data = pipe.stdout.read(w * h * 3)
        if not data:
            break
        i += 1
        if i > limit and limit != -1:
            continue
        if i > skip:
            yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=7, input_video_path=None, input_video_skip=0, error=None):
    """
    Render an animation. The supported output modes are:
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    titles = list(poses.keys())
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')

        ax.dist = 7.5
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:]
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents' definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    if 'Reconstruction' in titles[n]:
                        ax.set_title(
                            titles[n] + ' frame ' + str(i + 1) + ' P1 error ' + str(round(error[titles[n]][i], 2)) + 'mm')
                    else:
                        ax.set_title(titles[n] + ' frame ' + str(i + 1))
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1]:
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    if 'Reconstruction' in titles[n]:
                        ax.set_title(titles[n] + ' frame ' + str(i + 1) + ' P1 error ' + str(round(error[titles[n]][i], 2)) + 'mm')
                    else:
                        ax.set_title(titles[n] + ' frame ' + str(i + 1))
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

            points.set_offsets(keypoints[i])

        print('{}/{}      '.format(i, limit), end='\r')

    # fig.tight_layout()
    fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate, codec='mpeg4')
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        writer = PillowWriter(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, dpi=80, writer=writer)
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def render_3d_animation(poses, skeleton, fps, bitrate, azim, output,
                        limit=-1, size=6, reverse=False, error=None):
    """
    Render an animation. The supported output modes are:
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    titles = list(poses.keys())
    fig = plt.figure(figsize=(size * len(poses), size))

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.5
    for index, (title, data) in enumerate(poses.items()):
        limit = data.shape[0]
        ax = fig.add_subplot(1, 1 + len(poses), index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')

        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    initialized = False

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    if reverse:
                        ax_title = titles[n] + ' time ' + str(limit - i)
                        if error is not None:
                            ax_title = ax_title + ' error ' +  str(round(error[titles[n]][i], 2)) + 'mm'
                    else:
                        ax_title = titles[n] + ' time ' + str(i)

                    ax.set_title(ax_title)
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):
                    if reverse:
                        ax_title = titles[n] + ' time ' + str(limit - i)
                        if error is not None:
                            ax_title = ax_title + ' error ' + str(round(error[titles[n]][i], 2)) + 'mm'
                    else:
                        ax_title = titles[n] + ' time ' + str(i)

                    ax.set_title(ax_title)
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

        print('{}/{}      '.format(i, limit), end='\r')

    # fig.tight_layout()
    fig.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate, codec='mpeg4')
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        writer = PillowWriter(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, dpi=80, writer=writer)
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
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
    args.viz_output = 'viz.gif'
else:
    viz_output_list = args.viz_output.split('/')
    if len(viz_output_list) == 1:
        viz_path = os.path.join(args.checkpoint,
                                str(args.number_of_frames) + 'f',
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
                                              mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1,
                                              with_time_emb=params['with_time_emb'])

            model_diffusion = GaussianDiffusion(model=model_pos,
                                                timesteps=params['timesteps'],
                                                sampling_timesteps=params['sampling_timesteps'],
                                                loss_type='l2',
                                                clip_denoised=params['clip_denoised'],
                                                beta_schedule=params['beta_schedule'],
                                                ddim_sampling_eta=params['ddim_sampling_eta'])


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
            anim_reverse_diffusion_output = {}

            input_keypoints = []
            print('Rendering...')
            with torch.no_grad():
                model_diffusion.eval()
                for batch_id, (_, trajectory, inputs_3d, inputs_3d_norm, inputs_2d, inputs_2d_flip, target_mask, inputs_3d_frame_id, inputs_2D_frame_id, _, action, subject, _, _, cam_ind) in enumerate(viz_dataloader):
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
                        diffusion_3d_pos += trajectory.unsqueeze(-1).repeat(1,1,1,1,num_sample)
                        diffusion_3d_pos = diffusion_3d_pos.permute(4,1,2,3,0)[:,0,:,:,0]
                        diffusion_3d_pos = viz_dataloader.dataset.reverse_norm_3d_pose(diffusion_3d_pos)
                        anim_diffusion_output['Diffusion process'] = diffusion_3d_pos

                    inputs_3d_norm_flip = inputs_3d_norm.clone()
                    inputs_3d_norm_flip[:, :, :, 0] *= -1
                    inputs_3d_norm_flip[:, :, joints_left + joints_right] = inputs_3d_norm_flip[:, :,
                                                                            joints_right + joints_left]

                    # Predict 3D poses
                    _, predicted_3d_pos, reverse_diffusion_3d_pos, start_3d_pos_est = model_diffusion(clean_3d_pose=inputs_3d_norm,
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
                    error = mpjpe(predicted_3d_pos, inputs_3d, reduce='none')
                    if batch_id == 0:
                        inputs_3d_one_frame = inputs_3d[0, 0, :, :]

                        reverse_diffusion_3d_pos = viz_dataloader.dataset.reverse_norm_3d_pose(reverse_diffusion_3d_pos)
                        num_sample = reverse_diffusion_3d_pos.size(4)

                        inputs_3d_one_frame = inputs_3d_one_frame.unsqueeze(0).repeat(num_sample, 1, 1)
                        print('inputs_3d_one_frame {}'.format(inputs_3d_one_frame.size()))
                        print('reverse_diffusion_3d_pos {}'.format(reverse_diffusion_3d_pos.size()))
                        error_reverse_diffusion_tmp =  mpjpe(reverse_diffusion_3d_pos.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0],
                                                             inputs_3d_one_frame,
                                                             reduce='none')
                        error_reverse_diffusion_tmp = error_reverse_diffusion_tmp.cpu().numpy() * 1000
                        error_reverse_diffusion['Reverse diffusion process'] = error_reverse_diffusion_tmp

                        reverse_diffusion_3d_pos = reverse_diffusion_3d_pos.cpu()
                        reverse_diffusion_3d_pos += trajectory.unsqueeze(-1).repeat(1, 1, 1, 1, num_sample)
                        reverse_diffusion_3d_pos = reverse_diffusion_3d_pos.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0]
                        anim_reverse_diffusion_output['Reverse diffusion process'] = reverse_diffusion_3d_pos

                        start_3d_pos_est = viz_dataloader.dataset.reverse_norm_3d_pose(start_3d_pos_est)
                        num_sample = start_3d_pos_est.size(4)

                        error_reverse_diffusion_tmp = mpjpe(start_3d_pos_est.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0],
                                                            inputs_3d_one_frame,
                                                            reduce='none')
                        error_reverse_diffusion_tmp = error_reverse_diffusion_tmp.cpu().numpy() * 1000
                        error_reverse_diffusion['Start 3D pose est.'] = error_reverse_diffusion_tmp

                        start_3d_pos_est = start_3d_pos_est.cpu()
                        start_3d_pos_est += trajectory.unsqueeze(-1).repeat(1, 1, 1, 1, num_sample)
                        start_3d_pos_est = start_3d_pos_est.permute(4, 1, 2, 3, 0)[:, 0, :, :, 0]
                        anim_reverse_diffusion_output['Start 3D pose est.'] = start_3d_pos_est

                    predicted_3d_pos = predicted_3d_pos.cpu()
                    predicted_3d_pos += trajectory
                    predicted_3d_pos = predicted_3d_pos.view(-1, num_joints, 3)[target_mask == True, :, :]
                    anim_output['Reconstruction'].append(predicted_3d_pos)

                    error = error.cpu()
                    error = error.view(-1)[target_mask == True]
                    error_output['Reconstruction'].append(error)

                    inputs_3d =inputs_3d.cpu()
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
            error_output['Reconstruction'] = torch.cat(error_output['Reconstruction'], dim=0).numpy() * 1000
            print('error shape {}'.format(error_output['Reconstruction'].shape))
            anim_diffusion_output['Diffusion process'] = camera_to_world(anim_diffusion_output['Diffusion process'].numpy(), R=cam['orientation'],
                                                          t=cam['translation'])
            anim_reverse_diffusion_output['Reverse diffusion process'] = camera_to_world(
                anim_reverse_diffusion_output['Reverse diffusion process'].numpy(), R=cam['orientation'],
                t=cam['translation'])
            anim_reverse_diffusion_output['Start 3D pose est.'] = camera_to_world(
                anim_reverse_diffusion_output['Start 3D pose est.'].numpy(), R=cam['orientation'],
                t=cam['translation'])

            input_keypoints = torch.cat(input_keypoints, dim=0).numpy()
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

            render_animation(input_keypoints, keypoints_metadata, anim_output,
                             dataset.skeleton(), viz_fps, args.viz_bitrate, cam['azimuth'],
                             os.path.join(viz_path,args.viz_output),
                             limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                             input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                             input_video_skip=args.viz_skip, error=error_output)
            render_3d_animation(anim_diffusion_output,
                                dataset.skeleton(), 2, args.viz_bitrate, cam['azimuth'],
                                os.path.join(viz_path, 'diffusion_'+args.viz_output),
                                limit=args.viz_limit, size=args.viz_size)
            render_3d_animation(anim_reverse_diffusion_output,
                                dataset.skeleton(), 2, args.viz_bitrate, cam['azimuth'],
                                os.path.join(viz_path, 'reverse_diffusion_' + args.viz_output),
                                limit=args.viz_limit, size=args.viz_size, reverse=True,
                                error=error_reverse_diffusion)


