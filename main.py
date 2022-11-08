import numpy as np
from pyntcloud import PyntCloud
import copy
import scipy.stats as stats
import scipy.io as io
import plotly
import plotly.graph_objects as go
import scipy
from plotly.subplots import make_subplots

global X, Y, Z

def func(data, A, B, C, D, E):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    x, y = data.T

    return A + (B*x) + (C*y) + (D*x**2) + (E*x*y)

def func_array(data, A, B, C, D, E):
    # unpacking the multi-dim. array column-wise, that's why the transpose
    result = np.zeros(shape=(data.shape[0]))
    for row in range(0, data.shape[0]):
        x, y = data[row][0]
        result[row] = A + (B*x) + (C*y) + (D*x**2) + (E*x*y)
    return result

def print_fitted_params(fittedParameters):
    from string import ascii_uppercase
    for i, j in zip(fittedParameters, ascii_uppercase):
        print(f"{j} = {i:.3f}")

def compare_matlab_array(array_object, array_name):
    print("Compare %s: "%(array_name),  (io.loadmat("arrays/"+array_name)[array_name] == array_object).all())

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def plot_point_cloud(xs, ys, zs, xhats, yhats, zhats):
    trace1 = go.Scatter3d(
    x=xs, 
    y=ys, 
    z=zs, 
    marker=go.scatter3d.Marker(
        size=3,
        color='rgb(0,0,255)',  # set color to an array/list of desired values
        #colorscale='Viridis',   # choose a colorscale
        ), 
    opacity=0.8, 
    mode='markers'
    )

    trace2 = go.Scatter3d(
    x=xhats, 
    y=yhats, 
    z=zhats, 
    marker=go.scatter3d.Marker(
        size=3,
        color='rgb(255,0,0)',  # set color to an array/list of desired values
        #colorscale='Viridis',   # choose a colorscale
        ), 
    opacity=0.8, 
    mode='markers'
)

    fig = make_subplots()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    plotly.offline.plot(fig)

Pstat = np.ones((2, 3))
Pstat[:, 0] = [-330, 180]
Pstat[:, 1] = [-280, 200]
Pstat[:, 2] = [950, 1300]

compare_matlab_array(Pstat, "Pstat")

Pstat2 = np.ones((2, 3))
Pstat2[:, 0] = [-330, 180]
Pstat2[:, 1] = [-145, 110]
Pstat2[:, 2] = [950, 1300]
compare_matlab_array(Pstat2, "Pstat2")

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
for i in range(initPC, 41):
    ptCloud = PyntCloud.from_file(filename_prefix + str(i) + '.ply')
    ptCloud2 = copy.deepcopy(ptCloud)
    # Cropping
    
    location = np.array(ptCloud.points)[:, 0:3]
    indices = (location[:, 0] <= np.max(Pstat[:, 0])) & (location[:, 0] >= np.min(Pstat[:, 0])) & \
              (location[:, 1] <= np.max(Pstat[:, 1])) & (location[:, 1] >= np.min(Pstat[:, 1])) & \
              (location[:, 2] <= np.max(Pstat[:, 2])) & (location[:, 2] >= np.min(Pstat[:, 2]))
    matlab_indices = io.loadmat("arrays/indices")["indices"]
    matlab_indices = matlab_indices.reshape(1, matlab_indices.shape[0])[0]
    print("Compare indices: ",(indices==matlab_indices).all())


    indices = np.where(indices == 1)[0]
    matlab_indices_find = io.loadmat("arrays/indices_find")["indices"]
    matlab_indices_find = matlab_indices_find.reshape(1, matlab_indices_find.shape[0])[0]
    print("Compare indices_find: ",(indices+1==matlab_indices_find).all())

    ptCloud.apply_filter(indices)
    print("Compare point clouds after filter: ", 
    (np.array(ptCloud.points)[:, 0:3].astype("int") == io.loadmat("arrays/ptCloud_after_filter.mat")["ptCloud_after_filter"].astype("int")).all())

    # Cropping
    location = np.array(ptCloud2.points)[:, 0:3]
    indices = (location[:, 0] <= np.max(Pstat2[:, 0])) & (location[:, 0] >= np.min(Pstat2[:, 0])) & \
              (location[:, 1] <= np.max(Pstat2[:, 1])) & (location[:, 1] >= np.min(Pstat2[:, 1])) & \
              (location[:, 2] <= np.max(Pstat2[:, 2])) & (location[:, 2] >= np.min(Pstat2[:, 2]))
    indices = np.where(indices == 1)[0]
    ptCloud2.apply_filter(indices)
    print("Compare point clouds 2 after filter: ", 
    (np.array(ptCloud2.points)[:, 0:3].astype("int") == io.loadmat("arrays/ptCloud_after_filter2.mat")["ptCloud_after_filter2"].astype("int")).all())

    
    location = np.array(ptCloud2.points)[:, 0:3]
    X = location[:, 0]
    Y = location[:, 1]
    Z = location[:, 2]
    count_sensors = np.array(ptCloud2.points.shape[0])
    Inpt = np.round(stats.zscore(np.column_stack([X, Y])), decimals=4)

    # creating the numerical model for getting the sensors and displacements

    if jj==1:
        Inpthat = Inpt[SS, :]
        print(Inpt)
        fittedParameters, pcov = scipy.optimize.curve_fit(func,  Inpt, Z)
        print(print_fitted_params(fittedParameters=fittedParameters))
        Xhat = X[np.ravel(SS)]
        Yhat = Y[np.ravel(SS)]
        color = np.array(ptCloud2.points)[:, 3:6]
    else:
         fittedParameters, pcov = scipy.optimize.curve_fit(func, Inpt, Z)
    Zhat = func_array(Inpthat, fittedParameters[0],  fittedParameters[1],  fittedParameters[2],  fittedParameters[3],  fittedParameters[4])
    aux = [Xhat, Yhat, Zhat]
    print(aux)
    plot_point_cloud(X, Y, Z, Xhat, Yhat, Zhat)
    break

# here a non-linear surface fit is made with scipy's curve_fit()
# plot_point_cloud(X, Y, Z)