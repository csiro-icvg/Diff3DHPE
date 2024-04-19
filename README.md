# Diff3DHPE: A Diffusion Model for 3D Human Pose Estimation [R6D 2023]

<div style="text-align:center">
    <img src="assets/Diff3DHPE_MixSTE.png" width="1500" alt="Overall framework of Diff3DHPE during the reverse diffusion process in \textit{seq2seq} style. In the iteration step $t$, a 2D keypoint sequence $\tens{x}$ is concatenated with its corresponding noisy 3D predicted sequence $\hat{\tens{y}}_t$ along the channel dimension as the input $(\tens{x}, \hat{\tens{y}}_t)$. The backbone model takes $(\tens{x}, \hat{\tens{y}}_t)$ and $t$ to predict a final 3D sequence $\hat{\tens{y}}_{0,t}$ at the step $t$. Then, $\hat{\tens{y}}_{t-1}$ is obtained from a predefined reverse diffusion function and sent to the next iteration for refining. To note, the backbone model is MixSTE in this example."/>
</div>
The Pytroch implementation for <a href="https://openaccess.thecvf.com/content/ICCV2023W/R6D/html/Zhou_Diff3DHPE_A_Diffusion_Model_for_3D_Human_Pose_Estimation_ICCVW_2023_paper.html">"Diff3DHPE: A Diffusion Model for 3D Human Pose Estimation"</a>.

## Qualitative and quantitative results

<div style="text-align:center">
    <img src="assets/viz.gif" width="1200" height="400" />
</div>



### Human3.6M
#### CPN, 81 frames
|                       Method                       | MPJPE (mm) |
|:--------------------------------------------------:|:----------:|
| [PoseFormer](https://github.com/zczcwh/PoseFormer) |    44.3    |
| [MixSTE](https://github.com/JinluZhang1126/MixSTE) |    42.4    |
| [P-STMO-S](https://github.com/paTRICK-swk/P-STMO)  |    44.1    |
|                  Diff3DHPE-MixSTE                  |    42.0    |

#### CPN, 243 frames
|                       Method                       | MPJPE (mm) |
|:--------------------------------------------------:|:----------:|
| [MixSTE](https://github.com/JinluZhang1126/MixSTE) |    40.9    |
| [P-STMO-S](https://github.com/paTRICK-swk/P-STMO)  |    42.8    |
|                  Diff3DHPE-MixSTE                  |    40.0    |

#### GT, 81 frames
|                       Method                       | MPJPE (mm) |
|:--------------------------------------------------:|:----------:|
| [PoseFormer](https://github.com/zczcwh/PoseFormer) |    31.3    |
| [MixSTE](https://github.com/JinluZhang1126/MixSTE) |    25.9    |
|                  Diff3DHPE-MixSTE                  |    24.2    |

#### GT, 243 frames
|                       Method                       | MPJPE (mm) |
|:--------------------------------------------------:|:----------:|
| [MixSTE](https://github.com/JinluZhang1126/MixSTE) |    21.6    |
| [P-STMO-S](https://github.com/paTRICK-swk/P-STMO)  |    29.3    |
|                  Diff3DHPE-MixSTE                  |    20.2    |

### MPI-INF-3DHP
#### GT
|                       Method                       | Frames | PCK (%) | AUC (%) | MPJPE (mm) |
|:--------------------------------------------------:|:------:|:-------:|:-------:|:----------:|
| [PoseFormer](https://github.com/zczcwh/PoseFormer) |   9    |  88.6   |  56.4   |    77.1    |
| [MixSTE](https://github.com/JinluZhang1126/MixSTE) |   27   |  94.4   |  66.5   |    54.9    |
| [P-STMO-S](https://github.com/paTRICK-swk/P-STMO)  |   81   |  97.9   |  75.8   |    32.2    |
|                  Diff3DHPE-MixSTE                  |   27   |  99.1   |  84.8   |    19.6    |

## Environment
Please create the envirmennt by the following command:
```
conda env create -f Diff3DHPE.yml
```
## Dataset
Please refer to [data/README.MD](data/README.MD)

## Experiments
Please refer to [Experiments.sh](Experiments.sh)

Pretrained models will be available later.



## Visuzalization

### Figure
```
python visualization_fig.py --gpu_id 0 -sviz S9 -a "Photo 1" -cam 1 -s 81 -f 81 -b 1 --sampling_timesteps 9 -c checkpoint/h36m/ConditionalDiffusionMixSTES2SGRANDLinLift/cpn/f81 --evaluate ConditionalDiffusionMixSTES2SGRANDLinLift_l2_lr4e-4_useTembed_T_h36m_cpn_81f.bin --config configs/h36m_cpn_s2s_ConditionalDiffusionMixSTES2SGRANDLinLift.json --viz-video data/Videos/S9/Videos/Photo\ 1.55011271.mp4
```

### Animation
```
python visualization_ani.py --gpu_id 0 -sviz S9 -a "Photo 1" -cam 1 -s 81 -f 81 -b 4 --sampling_timesteps 9 -c checkpoint/h36m/ConditionalDiffusionMixSTES2SGRANDLinLift/cpn/f81 --evaluate ConditionalDiffusionMixSTES2SGRANDLinLift_l2_lr4e-4_useTembed_T_h36m_cpn_81f.bin --config configs/h36m_cpn_s2s_ConditionalDiffusionMixSTES2SGRANDLinLift.json --viz-video data/Videos/S9/Videos/Photo\ 1.55011271.mp4 --viz-output viz.mp4
```

## Citation
If you find this repo useful, please consider citing our paper:
```
@InProceedings{Zhou_2023_ICCV,
    author    = {Zhou, Jieming and Zhang, Tong and Hayder, Zeeshan and Petersson, Lars and Harandi, Mehrtash},
    title     = {Diff3DHPE: A Diffusion Model for 3D Human Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {2092-2102}
}
```

## Acknowledgement
Our code refers to the following repositories.
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [P-STMO-S](https://github.com/paTRICK-swk/P-STMO)