import numpy as np
from PointCloudBuilder import PointCloudBuilder
from PointCloudCompressor import PointCloudCompressor

pstat = np.ones((2, 3))
pstat[:, 0] = [-330, 180]
pstat[:, 1] = [-280, 200]
pstat[:, 2] = [950, 1300]

pstat2 = np.ones((2, 3))
pstat2[:, 0] = [-330, 180]
pstat2[:, 1] = [-145, 110]
pstat2[:, 2] = [950, 1300]

filename_prefix = 'pointcloud/master_2019OCT16_419pm'

avg_pc = 95000
n_acc = 90000
n_files = 270
init_pc = 40
components_number = 3
get_displacements = False


if get_displacements:
    point_cloud_builder = PointCloudBuilder(pstat, pstat2, filename_prefix, avg_pc, n_acc, n_files, init_pc)
    point_cloud_builder.run()
else:
    displacements = np.load("arrays/displacements.npy")
    x_hat = np.load("arrays/x_values.npy")
    y_hat = np.load("arrays/y_value.npy")
    z_variation = displacements[:, 2, :]
    pointCloudCompressor = PointCloudCompressor(z_variation, x_hat, y_hat, components_number)
    pointCloudCompressor.run()
