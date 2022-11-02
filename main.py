import numpy as np
from pyntcloud import PyntCloud
import copy
import scipy.stats as stats

Pstat = np.ones((2, 3))
Pstat[:, 0] = [-330, 180]
Pstat[:, 1] = [-280, 200]
Pstat[:, 2] = [950, 1300]

Pstat2 = np.ones((2, 3))
Pstat2[:, 0] = [-330, 180]
Pstat2[:, 1] = [-145, 110]
Pstat2[:, 2] = [950, 1300]

filename_prefix = 'pointcloud/master_2019OCT16_419pm'

avgPC = 95000
nAcc = 90000
SS = np.random.randint(1, avgPC, (nAcc, 1))

nFiles = 270
initPC = 40

jj = 1
ff = 1
ff2 = 4

Fs = 30

filename = 'LAGRARIAN444_' + str(nAcc) + 'SENSORS_KDE_' + str(Fs) + 'fps'
indices = 0
ptCloud = 0
for i in range(initPC, nFiles):
    ptCloud = PyntCloud.from_file(filename_prefix + str(i) + '.ply')
    ptCloud2 = copy.deepcopy(ptCloud)

    # Cropping
    location = np.array(ptCloud.points)[:, 0:3]
    indices = (location[:, 0] <= np.max(Pstat[:, 0])) & (location[:, 0] >= np.min(Pstat[:, 0])) & \
              (location[:, 1] <= np.max(Pstat[:, 1])) & (location[:, 1] >= np.min(Pstat[:, 1])) & \
              (location[:, 2] <= np.max(Pstat[:, 2])) & (location[:, 2] >= np.min(Pstat[:, 2]))
    
    print(indices.shape)
    indices = np.where(indices == 1)[0]
    ptCloud.apply_filter(indices)

    # Cropping
    location = np.array(ptCloud2.points)[:, 0:3]
    indices = (location[:, 0] <= np.max(Pstat2[:, 0])) & (location[:, 0] >= np.min(Pstat2[:, 0])) & \
              (location[:, 1] <= np.max(Pstat2[:, 1])) & (location[:, 1] >= np.min(Pstat2[:, 1])) & \
              (location[:, 2] <= np.max(Pstat2[:, 2])) & (location[:, 2] >= np.min(Pstat2[:, 2]))
    indices = np.where(indices == 1)[0]
    ptCloud2.apply_filter(indices)



    location = np.array(ptCloud2.points)[:, 0:3]
    X = location[:, 0]
    Y = location[:, 1]
    Z = location[:, 2]

    count_sensors = np.array(ptCloud2.points.shape[0])
    Inpt = stats.zscore(np.column_stack([X, Y]))

    # creating the numerical model for getting the sensors and displacements

    if jj==1:
        Inpthat = Inpt[SS, :]
        #print(Inpthat)

    #break'''
