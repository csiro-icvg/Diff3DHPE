## Dataset

The dataset put at here.

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. 

### Human3.6M

We use the processed data (2D GT keypoints: `data_2d_h36m_gt.npz`, 3D GT keypoints: `data_3d_h36m.npz`, and 2D CPN keypoints: `data_test_3dhp.npz`) provided by [Shan et al.](https://github.com/paTRICK-swk/P-STMO). Please refer to them to set up the Human3.6M dataset.

### MPI-INF-3DHP

We use the processed data (`data_train_3dhp.npz` and `data_test_3dhp.npz`) and evaluation script (`mpii_test_predictions_py.m`) provided by [Shan et al.](https://github.com/paTRICK-swk/P-STMO). Please refer to them to set up the MPI-INF-3DHP dataset.

### Directory Structure

Your directory structure should look like this

```
README.md
LICENCE
...
data/
  ├── h36m/
  | ├── S1/
  | | ├── MyPoses/
  | | └── StackedHourglass/
  | ├── S2/
  | ...
  | └── S11/
  ├── mpi_inf_3dhp/
  | ├── S1/
  | | ├── Seq1/
  | | └── Seq2/
  | ├── S2/
  | ...
  | └── S8/
  | ├── util/
  | └── mpi_inf_3dhp_test_set/
  |   ├── test_util/
  |   | ├── camera_calibration/
  |   | ...
  |   | └── mpii_test_predictions_py.m
  |   ├── TS1/
  |   | ├── imageSequence/
  |   | └── annot_data.mat
  |   ├── TS2/
  |   ...
  |   └── TS6/
  ├── data_train_3dhp.npz
  ├── data_test_3dhp.npz
  ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
  ├── data_2d_h36m_gt.npz
  └── data_3d_h36m.npz
```

