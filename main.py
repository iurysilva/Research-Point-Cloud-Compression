import numpy as np
from PointCloudBuilder import PointCloudBuilder

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


# point_cloud_builder = PointCloudBuilder(pstat, pstat2, filename_prefix, avg_pc, n_acc, n_files, init_pc)
# point_cloud_builder.run()

displacements = np.load("arrays/displacements.npy")
print(displacements.shape)